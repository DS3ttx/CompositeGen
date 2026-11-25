from miditok import REMI, TokenizerConfig

OUT_TOKEN = "OUT_Piano"


def make_tokenizer(config: Config):
    t_config = TokenizerConfig(**tokenizer_kwargs(config))
    return REMI(t_config)


def save_tokenizer(tok, path):
    tok.save(path)


def load_tokenizer(path):
    return REMI(params=Path(path))


def special_id(tok, name) -> int:
    for chave in (name + "_None", name):
        if chave in tok.vocab:
            return tok.vocab[chave]

    especiais = [word for word in tok.vocab if
                 word.endswith("_None") and not word.startswith(("Bar", "Pitch", "Position", "Duration"))]
    raise KeyError(f"token especial '{name}' não encontrado. Candidatos: {especiais} + OUT_*")


def encode_track(tok, score) -> list[int]:
    out = tok(score)
    seq = out[0] if isinstance(out, list) else out
    return list(seq.ids)


def build_sequence(tok, bass_ids, piano_ids):
    """[BOS] baixo [SEP] [OUT_Piano] piano [EOS]  ->  (ids, sep_idx)"""

    ids = [special_id(tok, "BOS")] + list(bass_ids)
    ids += [special_id(tok, "SEP"), special_id(tok, OUT_TOKEN)] + list(piano_ids) + [special_id(tok, "EOS")]
    sep_idx = 1 + len(bass_ids)
    return ids, sep_idx


def split_sequence(tok, ids, sep_idx):
    """Inverso de build_sequence: devolve (baixo, piano) sem os especiais."""
    ids = list(ids)
    bos, sep, out, eos = (special_id(tok, n) for n in ("BOS", "SEP", OUT_TOKEN, "EOS"))

    if sep_idx is None:
        sep_idx = ids.index(sep)

    assert ids[0] == bos and ids[sep_idx] == sep and ids[sep_idx + 1] == out

    bass = ids[1:sep_idx]
    piano = ids[sep_idx + 2:-1] if ids[-1] == eos else ids[sep_idx + 2:]
    return bass, piano


def decode_track(tok, ids, program, config, is_drum=False):
    specials = {special_id(tok, n) for n in config.special_tokens}
    ids = [i for i in ids if i not in specials]
    return tok([ids], programs=[(program, is_drum)])


def pitch_table(tok):
    """{id: pitch MIDI} e a inversa {pitch MIDI: id}."""
    table = {}
    for name, _id in tok.vocab.items():
        if name.startswith("Pitch_"):
            table[_id] = int(name.split("_")[1])

    inverse = {p: i for i, p in table.items()}
    return table, inverse


def transpose_ids(ids, k, table, inverse, pitch_range):
    """Soma k semitons só aos tokens de pitch. Retorna None se algum pitch sair da faixa."""
    low, high = pitch_range
    new = []
    for token in ids:
        if token in table:
            pitch = table[token] + k
            if not (low <= pitch < high) or pitch not in inverse:
                return None

            token = inverse[pitch]
        new.append(token)

    return new
