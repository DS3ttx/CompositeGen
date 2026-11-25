import hashlib
import numpy as np
from collections import Counter, defaultdict


def seq_hash(ids):
    return hashlib.md5(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def piano_hash(ids, sep):
    return seq_hash(ids[sep + 2:-1])  # sem OUT_Piano e sem EOS


def bass_hash(ids, sep):
    return seq_hash(ids[1:sep])  # sem BOS


def add_hashes(examples):
    for ex in examples:
        ex["h_full"] = seq_hash(ex["ids"])
        ex["h_piano"] = piano_hash(ex["ids"], ex["sep_idx"])
        ex["h_bass"] = bass_hash(ex["ids"], ex["sep_idx"])
    return examples


def dedup_within(examples, key="h_full"):
    """Mantém o primeiro exemplo de cada hash."""
    vistos = set()
    kept = []
    for ex in examples:
        if ex[key] not in vistos:
            vistos.add(ex[key])
            kept.append(ex)

    report = {"antes": len(examples), "depois": len(kept), "removidos": len(examples) - len(kept)}
    return kept, report


def classify_groups(examples):
    """Análise dos grupos de exemplos com o mesmo h_full."""
    groups = defaultdict(list)
    for ex in examples:
        groups[ex["h_full"]].append(ex)

    kinds = Counter()
    for g in groups.values():
        if len(g) < 2:
            continue
        if len({ex["track"] for ex in g}) > 1:
            kinds["músicas diferentes"] += 1
        elif len({(ex["track"], ex["window_idx"]) for ex in g}) == 1:
            kinds["mesma música e janela, stems diferentes"] += 1
        else:
            kinds["mesma música, janelas diferentes (repetição)"] += 1
    return dict(kinds)


def contaminated_tracks(examples, ref_hashes, key="h_full"):
    """Faixas que têm ao menos um exemplo cujo hash está em ref_hashes."""
    return {ex["track"] for ex in examples if ex[key] in ref_hashes}


def drop_tracks(examples, tracks):
    """Remove FAIXAS inteiras com janelas contaminadas."""
    return [ex for ex in examples if ex["track"] not in tracks]


def cross_split_check(a, b, key="h_full"):
    """Tamanho da interseção de hashes entre dois splits, deve ser 0."""
    return len({ex[key] for ex in a} & {ex[key] for ex in b})


def make_report(**etapas):
    return {nome: dict(valor) for nome, valor in etapas.items()}
