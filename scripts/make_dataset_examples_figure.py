from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)

# (archive_path, label)
EXAMPLES = [
    ("ffhq/test/R_FFHQ_17103.png", "Real\n(FFHQ)"),
    ("faceapp/train/F_FAP0_01256-3.png", "Fake\n(FaceApp)"),
    ("pggan_v1/test/F_PGN1_19431.png", "Fake\n(PGGAN v1)"),
    ("pggan_v2/test/F_PGN2_14439.png", "Fake\n(PGGAN v2)"),
    ("stargan/test/F_STGN_1715-19.png", "Fake\n(StarGAN)"),
    ("stylegan_celeba/test/F_SyCA_19417.png", "Fake\n(StyleGAN-CelebA)"),
    ("stylegan_ffhq/test/F_SyFQ_16808.png", "Fake\n(StyleGAN-FFHQ)"),
]


def main():
    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(12, 2.1))
    for ax, (archive_path, label) in zip(axes, EXAMPLES):
        img = Image.open(IMAGES_DIR / archive_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    fig.suptitle("Figure 1: One real image and one example from each fake source used in this project", fontsize=10)
    fig.tight_layout()
    out_path = OUT_DIR / "figure1_dataset_examples.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
