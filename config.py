import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # caminhos
    slakh_root: str = ""
    out_dir: str = ""

    # dados
    split_dirs: dict = field(default_factory=lambda: {"train": "train", "val": "validation", "test": "test"})
    input_class: str = "Bass"
    target_class: str = "Piano"
    n_bars: int = 8
    stride: int = 8  # saltos entre cada janela
    min_notes: int = 8  # por trilha e janela
    max_win_tokens: int = 768
    beats_per_bar: int = 4  # apenas 4/4
    limit: Optional[int] = None  # nmr maximo de faixas por split (debug)

    # tokenizador
    pitch_range: tuple = (21, 109)  # padrão do miditok
    beat_res: dict = field(default_factory=lambda: {"0-4": 4, "4-8": 2})
    use_velocities: bool = False
    use_programs: bool = False
    use_tempos: bool = False
    use_time_signatures: bool = False
    special_tokens: list = field(default_factory=lambda: ["PAD", "BOS", "EOS", "SEP", "OUT_Piano"])

    # augmentation
    max_transpose: int = 2
    seed: int = 7

    @classmethod
    def from_args(cls, args):
        cfg = cls(slakh_root=args.slakh, out_dir=args.out)

        if args.nbars is not None:
            cfg.n_bars = args.nbars
            cfg.stride = args.nbars

        if args.stride is not None:
            cfg.stride = args.stride

        if args.min_notes is not None:
            cfg.min_notes = args.min_notes

        if args.max_len is not None:
            cfg.max_win_tokens = args.max_len

        if args.limit is not None:
            cfg.limit = args.limit

        return cfg


def tokenizer_kwargs(config):
    """Converte o Config nos argumentos do TokenizerConfig do miditok"""
    beat_res = {}
    for key, value in config.beat_res.items():
        key_items = []
        for x in key.split("-"):
            key_items.append(int(x))

        beat_res[tuple(key_items)] = value

    return dict(
        pitch_range=tuple(config.pitch_range),
        beat_res=beat_res,
        use_velocities=config.use_velocities,
        use_programs=config.use_programs,
        use_tempos=config.use_tempos,
        use_time_signatures=config.use_time_signatures,
        special_tokens=list(config.special_tokens),
    )


def to_json(config, path):
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def from_json(path):
    with open(path) as f:
        data = json.load(f)

    data["pitch_range"] = tuple(data["pitch_range"])
    return Config(**data)
