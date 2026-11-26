# ==== Métricas extras: Pitch Entropy, Scale Consistency, Groove Consistency ====
# Usa as funções que já estão no notebook (TrainConfig, Vocab, build_model, generate_piano, set_seed).
# Não treina e não sobrescreve nada: gera de novo nos prompts de teste com os checkpoints salvos.
import json, time
from pathlib import Path
import numpy as np
import torch
from miditok import REMI

# ---- AJUSTE OS CAMINHOS ----
DATA_DIR = Path("/kaggle/input/SEU-DATASET/data")              # pasta com test.pt e tokenizer.json
CKPTS = {
    "transformer": "/kaggle/working/runs/transformer/best.pt",
    "lstm": "/kaggle/working/runs/lstm/best.pt",
    "rnn": "/kaggle/working/runs/rnn/best.pt",
}
N_PROMPTS = None          # None = todos os 49; ponha 3 para um teste rápido
OUT_JSON = "/kaggle/working/metricas_extras.json"

MAJOR, MINOR = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
SCALES = [{(root + i) % 12 for i in s} for root in range(12) for s in (MAJOR, MINOR)]


def notes_by_bar(ids, vocab, pos_of):
    """Lista de compassos; cada um com (posição, pitch) de cada nota."""
    bars, pos = [], 0
    for t in ids:
        if t == vocab.bar:
            bars.append([])
            pos = 0
        elif t in pos_of:
            pos = pos_of[t]
        elif t in vocab.pitch and bars:
            bars[-1].append((pos, vocab.pitch[t]))
    return bars


def pitch_entropy(pitches):
    """Entropia de Shannon (bits) do histograma de pitches."""
    if not pitches:
        return np.nan
    c = np.bincount(pitches, minlength=128).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def scale_consistency(pitches):
    """Maior fração de notas dentro de uma escala maior/menor (24 escalas)."""
    if not pitches:
        return np.nan
    pcs = [p % 12 for p in pitches]
    return max(sum(pc in s for pc in pcs) for s in SCALES) / len(pcs)


def groove_consistency(bars, n_pos):
    """1 - distância de Hamming média entre os padrões de ataque de compassos seguidos."""
    if len(bars) < 2:
        return np.nan
    grids = []
    for b in bars:
        g = np.zeros(n_pos, dtype=bool)
        for pos, _ in b:
            g[pos] = True
        grids.append(g)
    return float(1 - np.mean([np.sum(a != b) / n_pos for a, b in zip(grids[:-1], grids[1:])]))


def extra_metrics(piano_ids, vocab, pos_of, n_pos):
    bars = notes_by_bar(piano_ids, vocab, pos_of)
    pitches = [p for b in bars for _, p in b]
    return {"pitch_entropy": pitch_entropy(pitches),
            "scale_consistency": scale_consistency(pitches),
            "groove_consistency": groove_consistency(bars, n_pos)}


def mean_rows(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


# ---- dados ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = REMI(params=DATA_DIR / "tokenizer.json")
vocab = Vocab(tok)
pos_of = {i: int(n.split("_")[1]) for n, i in tok.vocab.items() if n.startswith("Position_")}
n_pos = len(pos_of)

test = torch.load(DATA_DIR / "test.pt", weights_only=False)
prompts = json.load(open(DATA_DIR / "test_prompts.json"))[:N_PROMPTS]
index = {(e["track"], e["window_idx"], e["piano_stem"]): e for e in test}
pairs = []
for p in prompts:
    e = index[(p["track"], p["window_idx"], p["piano_stem"])]
    pairs.append((e["ids"][1:e["sep_idx"]], e["ids"][e["sep_idx"] + 2:-1]))   # (baixo, piano real)

results = {"real": mean_rows([extra_metrics(real, vocab, pos_of, n_pos) for _, real in pairs])}

# ---- geração com cada checkpoint ----
for name, path in CKPTS.items():
    ckpt = torch.load(path, weights_only=False, map_location=device)
    cfg = TrainConfig(**ckpt["cfg"])
    model = build_model(cfg, vocab.size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    set_seed(cfg.seed)
    t0, rows = time.time(), []
    for bass, _ in pairs:
        gen, _ = generate_piano(model, vocab, bass, cfg, device)
        rows.append(extra_metrics(gen, vocab, pos_of, n_pos))
    results[name] = mean_rows(rows)
    print(f"{name}: {len(rows)} prompts em {time.time() - t0:.0f}s")

# ---- tabela ----
cols = ["real"] + list(CKPTS)
print(f"\n{'métrica':<20}" + "".join(f"{c:>14}" for c in cols))
for k in results["real"]:
    print(f"{k:<20}" + "".join(f"{results[c][k]:>14.3f}" for c in cols))
json.dump(results, open(OUT_JSON, "w"), indent=2)
print(f"\nsalvo em {OUT_JSON}")
