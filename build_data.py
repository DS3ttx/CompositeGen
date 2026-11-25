import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def build_parser():
    parser = argparse.ArgumentParser(description="Pré-processador do Slakh2100")
    parser.add_argument("--slakh", required=True, help="Caminho da raiz do dataset (slakh2100_flac_redux)")
    parser.add_argument("--out", required=True, help="Diretório de saída do pré-processamento")
    parser.add_argument("--nbars", type=int, help="Tamanho da janela em compassos")
    parser.add_argument("--stride", type=int, help="Salto entre janelas, em compassos")
    parser.add_argument("--min-notes", type=int, help="Quantidade mínima de notas por trilha em uma janela")
    parser.add_argument("--max-len", type=int, help="Máximo de tokens por exemplo")
    parser.add_argument("--limit", type=int, help="Nº máximo de faixas por split (debug)")
    parser.add_argument("--build-test", action="store_true", help="Constrói também o conjunto de teste")
    return parser


def build_split(config, tok, split):
    stats = Counter()
    examples = []

    for info in info_tracks(config, split, limit=config.limit, stats=stats):
        for w in iter_windows(info, config, stats):
            bass = encode_track(tok, w.bass)
            piano = encode_track(tok, w.piano)
            ids, sep = build_sequence(tok, bass, piano)

            if len(ids) > config.max_win_tokens:
                stats["maior_que_max_token"] += 1
                continue

            examples.append({"ids": ids, "sep_idx": sep, "track": w.track, "window_idx": w.window_idx,
                             "piano_stem": w.piano_stem, "qpm": w.qpm})

    return examples, stats


def pick_one_per_track(test, seed):
    """Um prompt por faixa, com semente fixa."""
    rng = random.Random(seed)
    per_track = {}
    for ex in test:
        per_track.setdefault(ex["track"], []).append(ex)

    prompts = []
    for track in sorted(per_track):
        ex = rng.choice(per_track[track])
        prompts.append({"track": track, "window_idx": ex["window_idx"], "piano_stem": ex["piano_stem"]})

    return prompts


def compute_stats(examples, tok, config):
    if not examples:
        return {"exemplos": 0}

    lens = np.array([len(ex["ids"]) for ex in examples])
    table, _ = pitch_table(tok)
    bar = tok.vocab.get("Bar_None")

    notas_baixo, notas_piano, vazios = [], [], 0
    for ex in examples:
        ids, sep = ex["ids"], ex["sep_idx"]
        baixo, piano = ids[1:sep], ids[sep + 2:-1]
        notas_baixo.append(sum(t in table for t in baixo))
        notas_piano.append(sum(t in table for t in piano))
        vazios += sum(1 for j, t in enumerate(piano)
                      if t == bar and (j + 1 == len(piano) or piano[j + 1] == bar))

    p50, p95, p99 = (float(v) for v in np.percentile(lens, [50, 95, 99]))
    return {
        "faixas": len({ex["track"] for ex in examples}),
        "exemplos": len(examples),
        "tokens_totais": int(lens.sum()),
        "comprimento": {"media": float(lens.mean()), "p50": p50, "p95": p95, "p99": p99, "max": int(lens.max())},
        "notas_por_janela_baixo": {"media": float(np.mean(notas_baixo)), "p5": float(np.percentile(notas_baixo, 5))},
        "notas_por_janela_piano": {"media": float(np.mean(notas_piano)), "p5": float(np.percentile(notas_piano, 5))},
        "fracao_compassos_vazios_piano": vazios / (len(examples) * config.n_bars),
    }


def build_train(config, tok):
    raw, st = build_split(config, tok, "train")
    train, rep = dedup_within(add_hashes(raw))
    return train, {"descartes": dict(st), "dedup": rep, "grupos_duplicados": classify_groups(raw)}


def build_val(config, tok, train):
    raw, st = build_split(config, tok, "val")
    val, rep_dedup = dedup_within(add_hashes(raw))

    h_tr = {ex["h_full"] for ex in train}
    cont = contaminated_tracks(val, h_tr)
    cont_piano = contaminated_tracks(val, {ex["h_piano"] for ex in train}, key="h_piano")
    cont_baixo = contaminated_tracks(val, {ex["h_bass"] for ex in train}, key="h_bass")

    n_antes, n_faixas = len(val), len({ex["track"] for ex in val})
    val = drop_tracks(val, cont)
    assert cross_split_check(train, val) == 0, "vazamento treino -> validação"

    return val, {"descartes": dict(st), "dedup": rep_dedup,
                 "contaminacao": {"faixas": n_faixas, "contaminadas_full": len(cont),
                                  "contaminadas_piano": len(cont_piano), "contaminadas_baixo": len(cont_baixo),
                                  "exemplos_antes": n_antes, "exemplos_depois": len(val)}}


def build_test(config, tok, train, val):
    raw, st = build_split(config, tok, "test")
    test, rep_dedup = dedup_within(add_hashes(raw))

    ref = {ex["h_full"] for ex in train} | {ex["h_full"] for ex in val}
    cont = contaminated_tracks(test, ref)
    n_antes, n_faixas = len(test), len({ex["track"] for ex in test})
    test = drop_tracks(test, cont)
    assert cross_split_check(train, test) == 0 and cross_split_check(val, test) == 0

    prompts = pick_one_per_track(test, config.seed)
    return test, prompts, {"descartes": dict(st), "dedup": rep_dedup,
                           "contaminacao": {"faixas": n_faixas, "contaminadas": len(cont),
                                            "exemplos_antes": n_antes, "exemplos_depois": len(test),
                                            "prompts": len(prompts)}}


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = Config.from_args(args)
    random.seed(config.seed)
    np.random.seed(config.seed)

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    to_json(config, out / "config.json")

    tok = make_tokenizer(config)
    save_tokenizer(tok, out / "tokenizer.json")
    print("vocabulário:", len(tok))

    report, stats = {}, {}

    train, report["train"] = build_train(config, tok)
    val, report["val"] = build_val(config, tok, train)
    splits = {"train": train, "val": val}

    if args.build_test:
        test, prompts, report["test"] = build_test(config, tok, train, val)
        splits["test"] = test
        with open(out / "test_prompts.json", "w") as f:
            json.dump(prompts, f, indent=2)

    for nome, exs in splits.items():
        torch.save(exs, out / f"{nome}.pt")
        stats[nome] = compute_stats(exs, tok, config)

    with open(out / "dedup_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2))

    return splits, tok, config
