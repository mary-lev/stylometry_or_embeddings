#!/usr/bin/env python3
"""
Reproducible regeneration of the Extended Waterfall figures from the
(corrected, re-run) residualization artifacts.

Produces, in results/figures/:
  - waterfall_figure.png       (Gemini text-embedding-001)
  - waterfall_figure_qwen.png  (Qwen3-8B)

Each figure has two panels (Russian, Italian) with 4 stages:
  Baseline -> After Interpretable Tiers -> After Char N-grams -> After Word Bigrams.
The final bar is the post-residualization residual (after the kernel control step),
matching Table 7's "Final residual" / "Final lift" rows and the abstract's headline
lifts. Values are read from the JSON artifacts, never hardcoded.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
FIGDIR = Path(__file__).parent.parent.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

STAGE_LABELS = ["Baseline", "After\nInterpretable\nTiers",
                "After\nChar N-grams", "After\nWord Bigrams"]
COLORS = ["#2b8ca4", "#a83279", "#e8920c", "#d4452a"]

# (file, experiment-block, interpretable-stage-key)
PANELS = {
    "gemini": {
        "Russian (29 authors)":  ("waterfall_extended_gemini.json",        "extended_6tier", "after_tier4"),
        "Italian (52 authors)":  ("waterfall_extended_italian_gemini.json", "extended_5tier", "after_tier3"),
    },
    "qwen": {
        "Russian (29 authors)":  ("waterfall_extended_qwen8b.json",          "extended_6tier", "after_tier4"),
        "Italian (52 authors)":  ("waterfall_extended_italian_qwen8b.json",   "extended_5tier", "after_tier3"),
    },
}


def load_panel(fname, block, interp_key):
    path = RESULTS / fname
    if not path.exists():
        # earlier runs wrote the Italian Qwen results with a hyphenated name
        alt = RESULTS / fname.replace("qwen8b", "qwen-8b")
        if alt.exists():
            path = alt
    d = json.load(open(path, encoding="utf-8"))
    exp = d["experiments"][block]
    st = exp["stages"]
    chance = exp.get("chance_level") or d.get("chance_level")
    keys = ["baseline", interp_key, "after_tier5", "after_kernel"]
    accs = [st[k]["accuracy"] * 100 for k in keys]
    lifts = [st[k]["accuracy"] / chance for k in keys]
    return accs, lifts, chance * 100


def draw(ax, accs, lifts, chance_pct, title):
    x = range(len(accs))
    bars = ax.bar(x, accs, color=COLORS, edgecolor="black", linewidth=0.8, width=0.7)
    top = max(accs)
    for i, (b, a, l) in enumerate(zip(bars, accs, lifts)):
        ax.text(b.get_x() + b.get_width() / 2, a + top * 0.02,
                f"{a:.1f}%\n({l:.1f}×)", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    # progressive-drop arrows between consecutive bar tops
    for i in range(len(accs) - 1):
        ax.annotate("", xy=(i + 1, accs[i + 1]), xytext=(i, accs[i]),
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=1.5))
        ax.text(i + 0.5, (accs[i] + accs[i + 1]) / 2 + top * 0.03,
                f"−{accs[i] - accs[i + 1]:.1f}pp", color="0.5",
                fontsize=10, ha="left", va="bottom")
    ax.axhline(chance_pct, ls="--", color="0.4", lw=1.5,
               label=f"Chance ({chance_pct:.1f}%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(0, top * 1.18)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)


def make_figure(model_key, out_name):
    panels = PANELS[model_key]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (title, (fname, block, interp)) in zip(axes, panels.items()):
        accs, lifts, ch = load_panel(fname, block, interp)
        draw(ax, accs, lifts, ch, title)
    fig.suptitle("Extended Waterfall: Progressive Residualization of Embeddings",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    print("Regenerating waterfall figures from corrected artifacts...")
    make_figure("gemini", "waterfall_figure.png")
    make_figure("qwen", "waterfall_figure_qwen.png")
    print("Done.")
