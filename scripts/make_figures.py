import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "report_assets"
FIG_DIR.mkdir(exist_ok=True)

MODEL_LABELS = {
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "xception": "XceptionNet",
}
COLORS = {
    "mobilenet_v3_large": "#4C72B0",
    "efficientnet_b0": "#55A868",
    "xception": "#C44E52",
}


def main():
    summary = json.load(open(ROOT / "runs" / "summary.json", encoding="utf-8"))
    rows = {r["model"]: r for r in summary["model_aggregates"]}
    gen_rows = {r["model"]: r for r in summary["generalisation_results"]}
    models = ["mobilenet_v3_large", "efficientnet_b0", "xception"]
    labels = [MODEL_LABELS[m] for m in models]
    colors = [COLORS[m] for m in models]

    # Figure 1: detection quality vs inference cost (mean +/- std over 3 seeds)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    f1s = [rows[m]["test_f1_mean"] * 100 for m in models]
    f1_err = [rows[m]["test_f1_std"] * 100 for m in models]
    recalls = [rows[m]["test_recall_mean"] * 100 for m in models]
    recall_err = [rows[m]["test_recall_std"] * 100 for m in models]
    x = range(len(models))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], f1s, width, yerr=f1_err, capsize=3, label="F1", color=colors)
    axes[0].bar([i + width / 2 for i in x], recalls, width, yerr=recall_err, capsize=3,
                label="Recall", alpha=0.55, color=colors)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Test set, %")
    lowest = min(v - e for v, e in zip(f1s + recalls, f1_err + recall_err))
    axes[0].set_ylim(max(0, min(95, lowest - 1)), 100)
    axes[0].set_title("(a) In-distribution detection quality\n(mean ± std, 3 seeds)")
    axes[0].legend(fontsize=8)

    infer_cost = [rows[m]["inference_emissions_kg_co2e_per_1000_mean"] * 1e6 for m in models]
    infer_cost_err = [rows[m]["inference_emissions_kg_co2e_per_1000_std"] * 1e6 for m in models]
    axes[1].bar(x, infer_cost, yerr=infer_cost_err, capsize=3, color=colors)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("mg CO2e per 1,000 images")
    axes[1].set_title("(b) Inference cost (mean ± std, 3 seeds)")
    for i, m in enumerate(models):
        axes[1].annotate(f"{rows[m]['num_params']/1e6:.1f}M params",
                          (i, infer_cost[i] + infer_cost_err[i]), ha="center", va="bottom", fontsize=7)

    fig.suptitle("Figure 2: Detection quality vs. deployment cost")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_quality_vs_cost.png", dpi=160)
    plt.close(fig)

    # Figure 2: generalisation gap
    fig, ax = plt.subplots(figsize=(5, 3.6))
    main_f1 = [rows[m]["test_f1_mean"] * 100 for m in models]
    gen_f1 = [gen_rows[m]["generalisation_f1"] * 100 for m in models]
    x = range(len(models))
    ax.bar([i - width / 2 for i in x], main_f1, width, label="In-distribution test F1", color=colors)
    ax.bar([i + width / 2 for i in x], gen_f1, width, label="Unseen-generator F1", color=colors, alpha=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("F1, %")
    ax.set_title("Figure 3: Generalisation to an unseen generator\n(StyleGAN-FFHQ held out of training)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_generalisation_gap.png", dpi=160)
    plt.close(fig)

    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
