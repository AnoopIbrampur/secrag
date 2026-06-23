"""Render eval results.json into a single figure for the README.

    PYTHONPATH=../src python plot_results.py --results results.json --out ../docs/eval.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_INK = "#1a1d24"
_ACCENT = "#4f9cf9"
_ACCENT2 = "#7bd389"
_MUTED = "#9aa0aa"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="../docs/eval.png")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())["summary"]
    r = data["retrieval"]
    k = data["k"]
    n = data["n_questions"]

    quality = {
        f"hit@{k}": r["hit"],
        f"soft hit@{k}": r["soft_hit"],
        f"precision@{k}": r["precision"],
        f"recall@{k}": r["recall"],
        "MRR": r["rr"],
        "faithfulness": data["faithfulness"],
        "correctness": data["correctness"],
    }
    L = data["latency_ms"]
    lat_groups = ["retrieval", "generation", "total"]
    p50 = [L["retrieval_p50"], L["generation_p50"], L["total_p50"]]
    p95 = [L["retrieval_p95"], L["generation_p95"], L["total_p95"]]

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": _MUTED,
                         "axes.labelcolor": _INK, "text.color": _INK,
                         "xtick.color": _INK, "ytick.color": _INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- quality metrics (0..1) ---
    labels = list(quality)
    vals = list(quality.values())
    colors = [_ACCENT] * 5 + [_ACCENT2, _ACCENT2]
    bars = ax1.barh(labels, vals, color=colors)
    ax1.set_xlim(0, 1.05)
    ax1.invert_yaxis()
    ax1.set_title(f"Retrieval & answer quality  (n={n}, k={k})", fontweight="bold")
    ax1.bar_label(bars, fmt="%.2f", padding=3, color=_INK)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # --- latency ---
    x = range(len(lat_groups))
    w = 0.38
    b1 = ax2.bar([i - w / 2 for i in x], p50, w, label="p50", color=_ACCENT)
    b2 = ax2.bar([i + w / 2 for i in x], p95, w, label="p95", color=_MUTED)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(lat_groups)
    ax2.set_ylabel("latency (ms)")
    ax2.set_title("Latency", fontweight="bold")
    ax2.legend(frameon=False)
    ax2.bar_label(b1, fmt="%.0f", padding=2)
    ax2.bar_label(b2, fmt="%.0f", padding=2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
