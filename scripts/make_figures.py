#!/usr/bin/env python3
"""Regenerate the two result figures of the manuscript from the recomputed numbers.

fig_missed_bugs.png : share of injected bugs missed per category, for LineGuard with
                      Claude Sonnet 5 (non-oracle), LineGuard with GPT-4o (oracle) and the
                      strongest analysis tool supporting the category.
fig_line_f1.png     : macro-averaged line-level F1-score per category for the same two
                      LineGuard configurations.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = json.load(open("reports/manuscript_numbers_keepblank.json"))
ORD = ["Re-entrancy", "Timestamp-Dependency", "Unchecked-Send", "Unhandled-Exceptions",
       "TOD", "Overflow-Underflow", "tx.origin"]
SHORT = ["Re-entrancy", "Timestamp\ndependency", "Unchecked\nsend", "Unhandled\nexceptions",
         "TOD", "Integer\noverflow", "tx.origin"]
# GPT-4o (oracle), re-scored on contracts 6-50: missed bugs and line-level F1
G_MISS = {"Re-entrancy": 451, "Timestamp-Dependency": 142, "Unchecked-Send": 15,
          "Unhandled-Exceptions": 349, "TOD": 441, "Overflow-Underflow": 465, "tx.origin": 97}
G_F1 = {"Re-entrancy": 0.728, "Timestamp-Dependency": 0.751, "Unchecked-Send": 0.986,
        "Unhandled-Exceptions": 0.825, "TOD": 0.522, "Overflow-Underflow": 0.791, "tx.origin": 0.918}
# strongest supporting tool per category (name, missed bugs), from Table 4
BEST = {"Re-entrancy": ("Slither", 0), "Timestamp-Dependency": ("Slither", 490),
        "Unchecked-Send": ("Mythril", 321), "Unhandled-Exceptions": ("Slither", 422),
        "TOD": ("Securify", 263), "Overflow-Underflow": ("Oyente", 814), "tx.origin": ("Slither", 0)}

x = np.arange(len(ORD)); w = 0.27
fig, ax = plt.subplots(figsize=(12, 4.2))
s = [100 * M[L]["missed"] / M[L]["bugs"] for L in ORD]
g = [100 * G_MISS[L] / M[L]["bugs"] for L in ORD]
t = [100 * BEST[L][1] / M[L]["bugs"] for L in ORD]
ax.bar(x - w, s, w, label="LineGuard, Claude Sonnet 5 (non-oracle)", color="#2f6f9f")
ax.bar(x, g, w, label="LineGuard, GPT-4o (oracle)", color="#8fb8d6")
ax.bar(x + w, t, w, label="strongest supporting tool", color="#c9c9c9")
for i, L in enumerate(ORD):
    ax.text(i + w, t[i] + 1.5, BEST[L][0], ha="center", fontsize=8, color="#444")
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
ax.set_ylabel("injected bugs missed (%)"); ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("paper/HighlightedVersion/fig_missed_bugs.png", dpi=200)

fig, ax = plt.subplots(figsize=(12, 4.0))
ax.bar(x - w / 2, [M[L]["F1"] for L in ORD], w, label="Claude Sonnet 5 (non-oracle)", color="#2f6f9f")
ax.bar(x + w / 2, [G_F1[L] for L in ORD], w, label="GPT-4o (oracle)", color="#8fb8d6")
for i, L in enumerate(ORD):
    ax.text(i - w / 2, M[L]["F1"] + .015, f"{M[L]['F1']:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, G_F1[L] + .015, f"{G_F1[L]:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
ax.set_ylabel("line-level F1-score"); ax.set_ylim(0, 1.12)
ax.legend(fontsize=9, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("paper/HighlightedVersion/fig_line_f1.png", dpi=200)
print("figures written to paper/HighlightedVersion/")
