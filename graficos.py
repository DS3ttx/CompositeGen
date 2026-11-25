"""Gráficos de comparação Transformer vs. LSTM (baixo -> piano).

Uso:
    unzip transformer_run.zip && unzip lstm_run.zip     # cria runs/transformer e runs/lstm
    python graficos.py --runs runs --out graficos [--exemplo Track01918_w1]

Precisa de: matplotlib, numpy, symusic (vem com o miditok).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---- paleta (2 séries validadas p/ daltonismo; real = neutro) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
BASS = "#c9c8c1"
COLORS = {"transformer": "#2a78d6", "lstm": "#eb6834", "real": "#52514e"}
NAMES = {"transformer": "Transformer", "lstm": "LSTM", "real": "Piano real"}
MODELS = ["transformer", "lstm"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def load(runs):
    hist = {m: json.load(open(runs / m / "history.json")) for m in MODELS}
    test = {m: json.load(open(runs / m / "test_metrics.json")) for m in MODELS}
    return hist, test


# ---- 1. curvas de aprendizado ----------------------------------------------------
def fig_curvas(hist, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    best_txt = {}
    for m in MODELS:
        h = hist[m]
        ep = [e["epoch"] for e in h]
        tr = [e["train"]["loss"] for e in h]
        va = [e["val"]["loss"] for e in h]
        acc = [e["val"]["acc"] for e in h]
        best = int(np.argmin(va))
        c = COLORS[m]
        a1.plot(ep, tr, color=c, ls=(0, (3, 2)), lw=1.5, alpha=0.7)
        a1.plot(ep, va, color=c)
        a1.plot(ep[best], va[best], "o", ms=7, color=c, mec=SURFACE, mew=2, zorder=5)
        best_txt[m] = f"{NAMES[m]}  (melhor val {va[best]:.3f}, ép. {ep[best]})"
        a2.plot(ep, acc, color=c)
        a2.annotate(f"{NAMES[m]} {acc[-1]:.1%}", (ep[-1], acc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    a1.set(title="Perda por época", xlabel="época", ylabel="cross-entropy (tokens do piano)")
    a1.set_ylim(top=max(hist["transformer"][1]["train"]["loss"], hist["lstm"][1]["train"]["loss"]) * 1.02)
    a1.legend(handles=[Line2D([], [], color=COLORS["transformer"], label=best_txt["transformer"]),
                       Line2D([], [], color=COLORS["lstm"], label=best_txt["lstm"]),
                       Line2D([], [], color=MUTED, label="validação  (● = melhor época)"),
                       Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=1.5, label="treino")],
              loc="upper right", fontsize=8.5)
    a2.set(title="Acurácia do próximo token (validação)", xlabel="época", ylabel="acurácia")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_xlim(right=max(len(hist[m]) for m in MODELS) + 6)
    a2.legend(handles=[Line2D([], [], color=COLORS[m], label=NAMES[m]) for m in MODELS],
              loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "1_curvas_aprendizado.png", dpi=200)
    plt.close(fig)


# ---- 2. métricas de teste --------------------------------------------------------
def fig_teste(test, out):
    specs = [("ppl", "Perplexidade no teste", "menor é melhor", "{:.2f}"),
             ("acc", "Acurácia no teste", "maior é melhor", "{:.1%}")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.4))
    for ax, (k, title, sub, fmt) in zip(axes, specs):
        vals = [test[m]["test"][k] for m in MODELS]
        y = np.arange(len(MODELS))[::-1]
        ax.barh(y, vals, height=0.55, color=[COLORS[m] for m in MODELS])
        for yi, v in zip(y, vals):
            ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=10, color=INK)
        ax.set_yticks(y, [NAMES[m] for m in MODELS])
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(f"{title}  ", loc="left")
        ax.text(1, 1.04, sub, transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
        ax.grid(False)
        ax.tick_params(axis="x", labelbottom=False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "2_teste.png", dpi=200)
    plt.close(fig)


# ---- 3. métricas da geração vs. piano real ---------------------------------------
def fig_geracao(test, out):
    specs = [("pc_sim_baixo", "Harmonia com o baixo", "similaridade de classes de altura", "{:.2f}", 1),
             ("vazios", "Compassos vazios", "fração por janela", "{:.0%}", 1),
             ("notas", "Notas por janela", "média", "{:.0f}", 1),
             ("pitch_medio", "Altura média", "MIDI (60 = Dó central)", "{:.1f}", 1)]
    rows = ["real"] + MODELS
    vals = {"real": test["transformer"]["real"], **{m: test[m]["gerado"] for m in MODELS}}

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    for ax, (k, title, sub, fmt, _) in zip(axes, specs):
        y = np.arange(len(rows))[::-1]
        ref = vals["real"][k]
        ax.axvline(ref, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        for yi, r in zip(y, rows):
            v = vals[r][k]
            ax.plot([ref, v], [yi, yi], color=GRID, lw=2, zorder=2)
            ax.plot(v, yi, "o", ms=9, color=COLORS[r], mec=SURFACE, mew=2, zorder=3)
            ax.annotate(fmt.format(v), (v, yi), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        lo = min(vals[r][k] for r in rows)
        hi = max(vals[r][k] for r in rows)
        pad = (hi - lo) * 0.35 or abs(hi) * 0.1
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.set_title(title)
        ax.set_xlabel(sub, fontsize=8.5, color=MUTED)
        ax.set_yticks(y, [NAMES[r] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        if k == "vazios":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle("Geração nos 49 prompts de teste: quanto mais perto da linha do piano real, melhor",
                 x=0.01, ha="left", fontsize=11, color=INK_2)
    fig.tight_layout()
    fig.savefig(out / "3_geracao_vs_real.png", dpi=200)
    plt.close(fig)


# ---- 4. piano roll de um exemplo ---------------------------------------------------
def notes_of(path):
    from symusic import Score
    s = Score(str(path))
    bar = 4 * s.ticks_per_quarter
    tracks = [[(n.time / bar, n.duration / bar, n.pitch) for n in t.notes] for t in s.tracks]
    return tracks[0], tracks[1]           # (baixo, piano) como gravado pelo train_kaggle.py


def fig_pianoroll(runs, out, exemplo):
    rows = [("real", runs / "transformer" / "midi" / f"{exemplo}_real.mid")]
    rows += [(m, runs / m / "midi" / f"{exemplo}_gerado.mid") for m in MODELS]
    data = [(r, *notes_of(p)) for r, p in rows]
    all_p = [n[2] for _, b, p in data for n in b + p]
    lo, hi = min(all_p) - 2, max(all_p) + 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, (r, bass, piano) in zip(axes, data):
        for notes, color in ((bass, BASS), (piano, COLORS[r])):
            for t, d, p in notes:
                ax.add_patch(plt.Rectangle((t, p - 0.4), max(d - 0.01, 0.02), 0.8,
                                           color=color, lw=0))
        ax.set_ylim(lo, hi)
        ax.set_title(NAMES[r], color=INK)
        for b in range(0, 10):
            ax.axvline(b, color=GRID, lw=1, zorder=0)
        ax.grid(False)
        ax.set_ylabel("pitch MIDI")
    axes[-1].set_xlim(0, 9)
    axes[-1].set_xticks(np.arange(9) + 0.5, [f"c{i + 1}" for i in range(9)])
    axes[-1].set_xlabel("compasso")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BASS, label="baixo (entrada)"),
                        plt.Rectangle((0, 0), 1, 1, color=COLORS["real"], label="piano real"),
                        plt.Rectangle((0, 0), 1, 1, color=COLORS["transformer"], label="piano Transformer"),
                        plt.Rectangle((0, 0), 1, 1, color=COLORS["lstm"], label="piano LSTM")],
               ncols=4, loc="upper right", fontsize=9)
    fig.suptitle(f"Mesmo baixo, três pianos — {exemplo}", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"4_pianoroll_{exemplo}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="graficos")
    ap.add_argument("--exemplo", default=None, help="ex.: Track01918_w1 (padrão: todos com MIDI)")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    hist, test = load(runs)
    fig_curvas(hist, out)
    fig_teste(test, out)
    fig_geracao(test, out)
    exemplos = [a.exemplo] if a.exemplo else sorted(
        p.name.removesuffix("_real.mid") for p in (runs / "transformer" / "midi").glob("*_real.mid"))
    for ex in exemplos:
        fig_pianoroll(runs, out, ex)
    print(f"{3 + len(exemplos)} figuras em {out.resolve()}")


if __name__ == "__main__":
    main()
