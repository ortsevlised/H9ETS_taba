import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GRADCAM_DIR = ROOT / "runs" / "gradcam" / "efficientnet_b0"
OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_PATH = GRADCAM_DIR / "gradcam_results.json"

SOURCE_LABELS = {
    "ffhq": "FFHQ",
    "pggan_v1": "PGGAN v1",
    "pggan_v2": "PGGAN v2",
    "stargan": "StarGAN",
    "stylegan_celeba": "StyleGAN-CelebA",
    "stylegan_ffhq": "StyleGAN-FFHQ",
}


def take(rows, used, preferred, fallback=None):
    for predicate in [preferred, fallback]:
        if predicate is None:
            continue
        for row in rows:
            key = row["output_image"]
            if key not in used and predicate(row):
                used.add(key)
                return row
    raise RuntimeError("Grad-CAM results do not contain enough examples for the six-image panel")


def select_examples(rows):
    used = set()
    correct_real = lambda r: r["correct"] and r["true_label"] == 0
    correct_fake = lambda r: r["correct"] and r["true_label"] == 1
    false_positive = lambda r: r["true_label"] == 0 and r["predicted_label"] == 1
    false_negative = lambda r: r["true_label"] == 1 and r["predicted_label"] == 0

    return [
        take(rows, used, correct_real),
        take(rows, used, lambda r: correct_fake(r) and r["source"] == "pggan_v1", correct_fake),
        take(rows, used, lambda r: correct_fake(r) and r["source"] == "stylegan_celeba", correct_fake),
        take(rows, used, false_positive),
        take(rows, used, lambda r: false_negative(r) and r["source"] == "stylegan_ffhq", false_negative),
        take(rows, used, lambda r: false_negative(r) and r["source"] == "pggan_v2", false_negative),
    ]


def caption(row):
    true_label = "Real" if row["true_label"] == 0 else "Fake"
    source = SOURCE_LABELS.get(row["source"], row["source"])
    if row["correct"]:
        outcome = "correct"
    elif row["true_label"] == 0:
        outcome = "FALSE POSITIVE"
    else:
        outcome = "FALSE NEGATIVE"
    return f"{true_label} ({source})\n{outcome}, p={row['predicted_prob']:.2f}"


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    examples = select_examples(rows)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    for ax, row in zip(axes.flat, examples):
        img = Image.open(ROOT / row["output_image"])
        ax.imshow(img)
        ax.set_title(caption(row), fontsize=9)
        ax.axis("off")

    fig.suptitle(
        "Figure 4: Grad-CAM examples (EfficientNet-B0, test split)\n"
        "top row: correct detections, bottom row: the two failure modes discussed in Section 5.4",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "figure4_gradcam_examples.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
