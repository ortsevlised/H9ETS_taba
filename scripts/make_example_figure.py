from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GRADCAM_DIR = ROOT / "runs" / "gradcam" / "efficientnet_b0"
OUT_DIR = ROOT / "report_assets"

# (filename, caption)
EXAMPLES = [
    ("R_FFHQ_19468__true0_pred0_p0.00.png", "Real (FFHQ)\ncorrect, p=0.00"),
    ("F_PGN1_13620__true1_pred1_p1.00.png", "Fake (PGGAN)\ncorrect, p=1.00"),
    ("F_SyCA_12235__true1_pred1_p1.00.png", "Fake (StyleGAN-CelebA)\ncorrect, p=1.00"),
    ("R_FFHQ_16569__true0_pred1_p0.89.png", "Real (FFHQ)\nFALSE POSITIVE, p=0.89"),
    ("F_SyFQ_18753__true1_pred0_p0.18.png", "Fake (StyleGAN-FFHQ)\nFALSE NEGATIVE, p=0.18"),
    ("F_PGN2_12658__true1_pred0_p0.45.png", "Fake (PGGAN v2)\nFALSE NEGATIVE, p=0.45"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    for ax, (filename, caption) in zip(axes.flat, EXAMPLES):
        img = Image.open(GRADCAM_DIR / filename)
        ax.imshow(img)
        ax.set_title(caption, fontsize=9)
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
