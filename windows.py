import bisect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from symusic import Score


@dataclass
class TrackInfo:
    track_dir: Path
    bass_stem: str
    bass_program: int
    piano_stems: list  # ["S02", "S05", ...]
    piano_programs: list


@dataclass
class Window:
    track: str
    window_idx: int
    piano_stem: str
    bass: Score
    piano: Score
    qpm: float


def stems_of(stems, cls):
    out = []
    for key, value in stems.items():
        if value["inst_class"] == cls:
            out.append(key)

    return sorted(out)


def info_tracks(config, split, limit=None, stats=None):
    root = Path(config.slakh_root) / config.split_dirs[split]
    out = []

    for track_dir in sorted(root.glob("Track*")):
        with open(track_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)

        stems = {key: value for key, value in meta["stems"].items() if value["midi_saved"]}
        bass = stems_of(stems, config.input_class)
        piano = stems_of(stems, config.target_class)

        if not bass or not piano:
            if stats is not None:
                stats["sem_par"] += 1
            continue

        out.append(TrackInfo(
            track_dir=track_dir,
            bass_stem=bass[0],
            bass_program=stems[bass[0]]["program_num"],
            piano_stems=piano,
            piano_programs=[stems[k]["program_num"] for k in piano]
        ))

        if limit is not None and len(out) >= limit:
            break

    return out


def load_pair(info, piano_stem):
    score_bass = Score(str(info.track_dir / "MIDI" / f"{info.bass_stem}.mid"))
    score_piano = Score(str(info.track_dir / "MIDI" / f"{piano_stem}.mid"))

    if len(score_bass.tracks) == 0 or len(score_piano.tracks) == 0:
        return None, "stem_vazio"

    if score_piano.ticks_per_quarter != score_bass.ticks_per_quarter:
        return None, "tpq_diferente"

    for t in score_bass.time_signatures:
        if t.numerator != 4 or t.denominator != 4:
            return None, "nao_4_4"

    return (score_bass, score_piano), None


def window_starts(score_bass, score_piano, config):
    bar = config.beats_per_bar * score_bass.ticks_per_quarter
    last = max(
        max((n.time for n in score_bass.tracks[0].notes), default=0),  # tempo da última nota de sb
        max((n.time for n in score_piano.tracks[0].notes), default=0)  # tempo da última nota de sp
    )
    n = int(last // bar) - config.n_bars + 1

    return [i * bar for i in range(0, n, config.stride)]


def note_times(score):
    return np.sort(np.array([n.time for n in score.tracks[0].notes], dtype=np.int64))


def count_in(times, s, e):
    # equivale a contar notas com s <= time < e
    return int(np.searchsorted(times, e, side="left") - np.searchsorted(times, s, side="left"))


def tempo_list(score):
    out = []
    for t in score.tempos:
        qpm = getattr(t, "qpm", None)
        if qpm is None:  # fallback se só existir mspq
            qpm = 60_000_000 / t.mspq
        out.append((t.time, float(qpm)))
    return sorted(out)


def tempo_at(tempos, tick, default=120.0):
    if not tempos:
        return default
    i = bisect.bisect_right([t for t, _ in tempos], tick) - 1
    return tempos[max(i, 0)][1]


def iter_windows(info, config, stats):
    for piano_stem in info.piano_stems:
        par, reason = load_pair(info, piano_stem)

        if par is None:
            stats[reason] += 1
            continue

        score_bass, score_piano = par
        time_bass, time_piano = note_times(score_bass), note_times(score_piano)
        tempos = tempo_list(score_bass)
        bar = config.beats_per_bar * score_bass.ticks_per_quarter

        for idx, start in enumerate(window_starts(score_bass, score_piano, config)):
            end = start + config.n_bars * bar
            if (count_in(time_bass, start, end) < config.min_notes
                    or count_in(time_piano, start, end) < config.min_notes):
                stats["poucas_notas"] += 1
                continue

            stats["janelas_ok"] += 1
            yield Window(
                track=info.track_dir.name,
                window_idx=idx,
                piano_stem=piano_stem,
                bass=score_bass.clip(start, end).shift_time(-start),
                piano=score_piano.clip(start, end).shift_time(-start),
                qpm=tempo_at(tempos, start),
            )
