# %% [markdown]
# # Treino: baixo -> piano (Transformer decoder-only vs. LSTM vs. RNN)
#
# Lê os artefatos do pré-processamento (train.pt, val.pt, test.pt, tokenizer.json,
# test_prompts.json) e treina o modelo escolhido em CONFIG.model.
#
# No Kaggle:
#   1. Suba o data.zip como Dataset (Add Input) e ative a GPU (T4 ou P100).
#   2. Primeira célula:  !pip install -q miditok==3.1.0
#   3. Cole este arquivo em células (os marcadores "# %%" separam) ou rode com
#      !python train_kaggle.py --model transformer   (ou lstm, rnn)

# %% imports
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from miditok import REMI
from symusic import Tempo


# %% configuração
@dataclass
class TrainConfig:
    model: str = "transformer"          # "transformer", "lstm" ou "rnn"
    data_dir: str = ""                  # vazio = procura train.pt em /kaggle/input
    out_dir: str = "/kaggle/working/runs"

    # transformer (~10.6M params)
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.15

    # lstm (~10.0M params, comparável)
    lstm_hidden: int = 640
    lstm_layers: int = 3
    lstm_dropout: float = 0.3

    # rnn vanilla (tanh) (~10.1M params, comparável)
    rnn_hidden: int = 1280
    rnn_layers: int = 3
    rnn_dropout: float = 0.3

    # otimização
    batch_size: int = 32
    lr: float = 0.0                     # 0 = padrão do modelo (3e-4 transformer, 1e-3 lstm, 5e-4 rnn)
    weight_decay: float = 0.1
    warmup_steps: int = 500
    epochs: int = 40
    patience: int = 6                   # early stopping (épocas sem melhorar val)
    grad_clip: float = 1.0
    max_transpose: int = 5              # augmentation em semitons (o pré-proc usava 2)
    num_workers: int = 2
    seed: int = 7
    max_len: int = 768

    # geração
    temperature: float = 1.0
    top_p: float = 0.95
    max_bars: int = 9                   # 8 compassos + 1: a quantização joga notas do fim
                                        # da janela p/ o tempo 1 de um 9º compasso (~14% dos dados)
    n_midi: int = 10                    # quantos MIDIs de exemplo salvar

    # debug: limita nº de exemplos e passos (0 = sem limite)
    debug_examples: int = 0
    debug_steps: int = 0


DEFAULT_LR = {"transformer": 3e-4, "lstm": 1e-3, "rnn": 5e-4}


def find_data_dir(cfg):
    if cfg.data_dir:
        return Path(cfg.data_dir)
    for pattern in ("/kaggle/input/**/train.pt", "./data/train.pt"):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError("train.pt não encontrado; defina data_dir")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %% tokenizador e dados
class Vocab:
    """Ids especiais e tabela de pitch a partir do tokenizador salvo."""

    def __init__(self, tok):
        self.tok = tok
        self.size = len(tok)
        self.id2name = {i: n for n, i in tok.vocab.items()}

        def sid(name):
            for key in (name + "_None", name):
                if key in tok.vocab:
                    return tok.vocab[key]
            raise KeyError(name)

        self.pad, self.bos, self.eos = sid("PAD"), sid("BOS"), sid("EOS")
        self.sep, self.out, self.bar = sid("SEP"), sid("OUT_Piano"), sid("Bar")
        self.specials = [self.pad, self.bos, self.eos, self.sep, self.out]
        self.pitch = {i: int(n.split("_")[1]) for n, i in tok.vocab.items() if n.startswith("Pitch_")}
        self.pitch_inv = {p: i for i, p in self.pitch.items()}


def transpose_ids(ids, k, vocab, pitch_range=(21, 109)):
    lo, hi = pitch_range
    out = []
    for t in ids:
        if t in vocab.pitch:
            p = vocab.pitch[t] + k
            if not (lo <= p < hi) or p not in vocab.pitch_inv:
                return None
            t = vocab.pitch_inv[p]
        out.append(t)
    return out


class PairDataset(Dataset):
    def __init__(self, examples, vocab, max_transpose=0):
        self.examples, self.vocab, self.max_transpose = examples, vocab, max_transpose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        ids = ex["ids"]
        if self.max_transpose > 0:
            for _ in range(3):
                k = random.randint(-self.max_transpose, self.max_transpose)
                if k == 0:
                    break
                new = transpose_ids(ids, k, self.vocab)
                if new is not None:
                    ids = new
                    break
        return torch.tensor(ids, dtype=torch.long), ex["sep_idx"]


def collate(batch, pad_id):
    lengths = torch.tensor([len(ids) for ids, _ in batch])
    B, L = len(batch), int(lengths.max())
    x = torch.full((B, L), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((B, L), dtype=torch.bool)
    for i, (ids, sep) in enumerate(batch):
        x[i, : len(ids)] = ids
        loss_mask[i, sep + 2 : len(ids)] = True      # piano + EOS
    return {"x": x, "loss_mask": loss_mask}


def worker_init_fn(_):
    random.seed(torch.initial_seed() % 2**32)


def make_loader(examples, vocab, cfg, train):
    ds = PairDataset(examples, vocab, cfg.max_transpose if train else 0)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers,
                      collate_fn=partial(collate, pad_id=vocab.pad), worker_init_fn=worker_init_fn,
                      pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=cfg.num_workers > 0)


# %% modelos
class Rotary(nn.Module):
    def __init__(self, head_dim, max_len, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

    def forward(self, x):                                  # (B, H, L, D)
        L = x.shape[2]
        cos, sin = self.cos[:, :, :L].to(x.dtype), self.sin[:, :, :L].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.h, self.p = h, p
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d), nn.Dropout(p))
        self.drop = nn.Dropout(p)

    def forward(self, x, rope):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.mlp(self.ln2(x))


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = Rotary(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                    for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight                 # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():               # escala residual (GPT-2)
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.drop(self.emb(x))
        for b in self.blocks:
            h = b(h, self.rope)
        return self.head(self.ln(h))


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        H = cfg.lstm_hidden
        self.emb = nn.Embedding(vocab_size, H)
        self.drop = nn.Dropout(cfg.lstm_dropout)
        self.lstm = nn.LSTM(H, H, num_layers=cfg.lstm_layers, batch_first=True,
                            dropout=cfg.lstm_dropout)
        self.head = nn.Linear(H, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h, _ = self.lstm(self.drop(self.emb(x)))
        return self.head(self.drop(h))


class DecoderRNN(nn.Module):
    """RNN "normal" (Elman, tanh). Mesma estrutura do DecoderLSTM, sem portões."""

    def __init__(self, vocab_size, cfg):
        super().__init__()
        H = cfg.rnn_hidden
        self.emb = nn.Embedding(vocab_size, H)
        self.drop = nn.Dropout(cfg.rnn_dropout)
        self.rnn = nn.RNN(H, H, num_layers=cfg.rnn_layers, nonlinearity="tanh",
                          batch_first=True, dropout=cfg.rnn_dropout)
        self.head = nn.Linear(H, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)
        for name, p in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)          # ajuda a RNN a não explodir/sumir o gradiente
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        return self.head(self.drop(h))


def build_model(cfg, vocab_size):
    if cfg.model == "transformer":
        return DecoderTransformer(vocab_size, cfg)
    if cfg.model == "lstm":
        return DecoderLSTM(vocab_size, cfg)
    if cfg.model == "rnn":
        return DecoderRNN(vocab_size, cfg)
    raise ValueError(cfg.model)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# %% treino
def amp_settings(device):
    if device.type != "cuda":
        return False, torch.float32
    # T4/P100 não têm bf16 nativo -> fp16 + GradScaler
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return True, torch.bfloat16
    return True, torch.float16


def make_optimizer(model, cfg, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())


def make_scheduler(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))   # cosseno até 10% do lr
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def run_epoch(model, loader, device, amp, opt=None, sched=None, scaler=None, cfg=None):
    train = opt is not None
    model.train(train)
    use_amp, amp_dtype = amp
    tot_loss, tot_tok, tot_correct = 0.0, 0, 0

    for step, batch in enumerate(loader):
        if cfg is not None and cfg.debug_steps and step >= cfg.debug_steps:
            break
        x = batch["x"].to(device, non_blocking=True)
        m = batch["loss_mask"].to(device, non_blocking=True)
        inp, tgt, mm = x[:, :-1], x[:, 1:], m[:, 1:]

        with torch.set_grad_enabled(train), \
             torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(inp)
        logits = logits.float()[mm]
        target = tgt[mm]
        loss_sum = F.cross_entropy(logits, target, reduction="sum")
        n = int(mm.sum())

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss_sum / n).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

        tot_loss += loss_sum.item()
        tot_tok += n
        tot_correct += int((logits.argmax(-1) == target).sum())

    loss = tot_loss / max(1, tot_tok)
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "acc": tot_correct / max(1, tot_tok)}


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = find_data_dir(cfg)
    run_dir = Path(cfg.out_dir) / cfg.model
    run_dir.mkdir(parents=True, exist_ok=True)
    lr = cfg.lr or DEFAULT_LR[cfg.model]

    tok = REMI(params=data_dir / "tokenizer.json")
    vocab = Vocab(tok)
    train_ex = torch.load(data_dir / "train.pt", weights_only=False)
    val_ex = torch.load(data_dir / "val.pt", weights_only=False)
    if cfg.debug_examples:
        train_ex, val_ex = train_ex[: cfg.debug_examples], val_ex[: cfg.debug_examples]

    train_dl = make_loader(train_ex, vocab, cfg, train=True)
    val_dl = make_loader(val_ex, vocab, cfg, train=False)

    model = build_model(cfg, vocab.size).to(device)
    amp = amp_settings(device)
    steps_per_epoch = min(len(train_dl), cfg.debug_steps or len(train_dl))
    total = cfg.epochs * steps_per_epoch
    opt = make_optimizer(model, cfg, lr)
    sched = make_scheduler(opt, min(cfg.warmup_steps, total // 10 + 1), total)
    scaler = torch.amp.GradScaler(device.type, enabled=amp[0] and amp[1] == torch.float16)

    print(f"dados: {data_dir} | treino {len(train_ex)} | val {len(val_ex)} | vocab {vocab.size}")
    print(f"modelo: {cfg.model} | {n_params(model) / 1e6:.2f}M params | device {device} | "
          f"amp {amp[1] if amp[0] else 'off'} | lr {lr} | {steps_per_epoch} passos/época")

    history, best, bad = [], float("inf"), 0
    ckpt_path = run_dir / "best.pt"
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, device, amp, opt, sched, scaler, cfg)
        with torch.no_grad():
            va = run_epoch(model, val_dl, device, amp, cfg=cfg)
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": sched.get_last_lr()[0]})
        flag = ""
        if va["loss"] < best:
            best, bad, flag = va["loss"], 0, "  *melhor*"
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch,
                        "val": va}, ckpt_path)
        else:
            bad += 1
        print(f"época {epoch:3d} | treino {tr['loss']:.4f} (acc {tr['acc']:.3f}) | "
              f"val {va['loss']:.4f} ppl {va['ppl']:.2f} acc {va['acc']:.3f} | "
              f"{time.time() - t0:.0f}s{flag}")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        if bad >= cfg.patience:
            print(f"early stopping: {cfg.patience} épocas sem melhorar")
            break

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"melhor época {ckpt['epoch']}: val loss {ckpt['val']['loss']:.4f}, ppl {ckpt['val']['ppl']:.2f}")
    return model, vocab, data_dir, run_dir, history


# %% geração
@torch.no_grad()
def generate_piano(model, vocab, bass_ids, cfg, device):
    """Gera o piano para um baixo. Para em EOS ou ao tentar abrir o compasso max_bars+1."""
    model.eval()
    seq = [vocab.bos] + list(bass_ids) + [vocab.sep, vocab.out]
    start = len(seq)
    banned = torch.tensor([vocab.pad, vocab.bos, vocab.sep, vocab.out], device=device)
    n_bars = 0
    while len(seq) < cfg.max_len:
        x = torch.tensor(seq, device=device)[None]
        logits = model(x)[0, -1].float() / cfg.temperature
        logits[banned] = -float("inf")
        probs = F.softmax(logits, -1)
        sp, si = probs.sort(descending=True)
        sp[(sp.cumsum(-1) - sp) > cfg.top_p] = 0             # nucleus (top-p)
        nxt = int(si[torch.multinomial(sp / sp.sum(), 1)])
        if nxt == vocab.bar:
            if n_bars == cfg.max_bars:
                nxt = vocab.eos
            else:
                n_bars += 1
        seq.append(nxt)
        if nxt == vocab.eos:
            break
    piano = seq[start:]
    ended = bool(piano) and piano[-1] == vocab.eos
    return (piano[:-1] if ended else piano), ended


# %% métricas simbólicas
def bars_of(ids, vocab):
    """Lista de compassos; cada um é a lista de pitches MIDI daquele compasso."""
    bars = []
    for t in ids:
        if t == vocab.bar:
            bars.append([])
        elif t in vocab.pitch and bars:
            bars[-1].append(vocab.pitch[t])
    return bars


def pc_hist(pitches):
    h = np.zeros(12)
    for p in pitches:
        h[p % 12] += 1
    return h


def window_metrics(bass_ids, piano_ids, vocab, max_bars):
    bb, pb = bars_of(bass_ids, vocab), bars_of(piano_ids, vocab)
    sims = []
    for i in range(min(len(bb), len(pb))):
        a, b = pc_hist(bb[i]), pc_hist(pb[i])
        if a.sum() and b.sum():
            sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    notes = sum(len(b) for b in pb)
    return {
        "compassos_ok": float(len(pb) <= max_bars),
        "vazios": float(np.mean([len(b) == 0 for b in pb])) if pb else 1.0,
        "notas": notes,
        "pc_sim_baixo": float(np.mean(sims)) if sims else 0.0,
        "pitch_medio": float(np.mean([p for b in pb for p in b])) if notes else 0.0,
    }


def summarize(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def to_midi(vocab, bass_ids, piano_ids, qpm, path):
    tok = vocab.tok
    score = tok([list(bass_ids)], programs=[(33, False)])
    piano = tok([list(piano_ids)], programs=[(0, False)])
    score.tracks.append(piano.tracks[0])
    score.tempos = [Tempo(0, float(qpm))]
    score.dump_midi(str(path))


def evaluate_test(model, vocab, data_dir, run_dir, cfg):
    device = next(model.parameters()).device
    test = torch.load(data_dir / "test.pt", weights_only=False)
    with open(data_dir / "test_prompts.json") as f:
        prompts = json.load(f)
    index = {(e["track"], e["window_idx"], e["piano_stem"]): e for e in test}

    # perplexidade no teste (todos os exemplos)
    amp = amp_settings(device)
    with torch.no_grad():
        ppl = run_epoch(model, make_loader(test, vocab, cfg, train=False), device, amp, cfg=cfg)
    print(f"teste: loss {ppl['loss']:.4f} ppl {ppl['ppl']:.2f} acc {ppl['acc']:.3f}")

    midi_dir = run_dir / "midi"
    midi_dir.mkdir(exist_ok=True)
    gen_rows, real_rows, eos_ok = [], [], 0
    set_seed(cfg.seed)
    t0 = time.time()
    for j, p in enumerate(prompts):
        ex = index[(p["track"], p["window_idx"], p["piano_stem"])]
        ids, sep = ex["ids"], ex["sep_idx"]
        bass, real = ids[1:sep], ids[sep + 2 : -1]
        gen, ended = generate_piano(model, vocab, bass, cfg, device)
        eos_ok += ended
        gen_rows.append(window_metrics(bass, gen, vocab, cfg.max_bars))
        real_rows.append(window_metrics(bass, real, vocab, cfg.max_bars))
        if j < cfg.n_midi:
            name = f"{p['track']}_w{p['window_idx']}"
            to_midi(vocab, bass, gen, ex["qpm"], midi_dir / f"{name}_gerado.mid")
            to_midi(vocab, bass, real, ex["qpm"], midi_dir / f"{name}_real.mid")

    g, r = summarize(gen_rows), summarize(real_rows)
    print(f"\ngeração: {len(prompts)} prompts em {time.time() - t0:.0f}s | "
          f"terminou com EOS: {eos_ok}/{len(prompts)}")
    print(f"{'métrica':<16}{'gerado':>10}{'real':>10}")
    for k in g:
        print(f"{k:<16}{g[k]:>10.3f}{r[k]:>10.3f}")
    result = {"test": ppl, "gerado": g, "real": r, "eos": eos_ok / len(prompts)}
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMIDIs em {midi_dir}")
    return result


# %% execução
def parse_cli():
    ap = argparse.ArgumentParser()
    for name, default in asdict(TrainConfig()).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args, _ = ap.parse_known_args()     # parse_known_args: ignora args do Jupyter
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    in_notebook = "ipykernel" in sys.modules
    cfg = TrainConfig() if in_notebook else parse_cli()
    # No notebook, edite aqui, ex.: cfg = TrainConfig(model="lstm")
    model, vocab, data_dir, run_dir, history = train(cfg)
    evaluate_test(model, vocab, data_dir, run_dir, cfg)
