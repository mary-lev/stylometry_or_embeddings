#!/usr/bin/env python3
"""Binary attribution error-by-length figure, plotted from the per-poem
method-comparison analysis (method_differences_*.json) so it is consistent
with Table 13, the case studies (Section 8), and the pairwise means in Section 5.3."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def load(f):
    d = json.load(open(RES / f))
    la = d["length_analysis"]
    labels = [b["label"] for b in la]
    emb = [b["emb_error"] * 100 for b in la]
    sty = [b["stylo_error"] * 100 for b in la]
    ns  = [b["n_poems"] for b in la]
    tot = sum(ns)
    oe = sum(b["n_poems"] * b["emb_error"] for b in la) / tot
    os_ = sum(b["n_poems"] * b["stylo_error"] for b in la) / tot
    return labels, emb, sty, ns, (1 - oe) * 100, (1 - os_) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (f, title) in zip(axes, [("method_differences_gemini.json", "Russian (Binary Classification)"),
                                  ("method_differences_italian.json", "Italian (Binary Classification)")]):
    labels, emb, sty, ns, oe, os_ = load(f)
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, emb, w, label="Embeddings", color="#3b6fb5", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, sty, w, label="Stylometry", color="#e07b39", edgecolor="black", linewidth=0.6)
    for i, n in enumerate(ns):
        ax.text(i, max(emb[i], sty[i]) + 0.3, f"n={n}", ha="center", va="bottom", fontsize=8, color="0.4")
    ax.text(0.02, 0.97, f"Overall: Emb={oe:.1f}%, Styl={os_:.1f}%", transform=ax.transAxes,
            va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Error Rate (%)"); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "binary_fig2_error_by_length.png", dpi=200, bbox_inches="tight")
print("saved", OUT / "binary_fig2_error_by_length.png")
