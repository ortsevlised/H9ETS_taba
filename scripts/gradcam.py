import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch.utils.data import DataLoader

from common import IMAGES_DIR, ManifestDataset, build_model, eval_transform

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests" / "main_manifest.csv"
OUT_DIR = ROOT / "runs" / "gradcam"
SEED = 123
N_PER_CLASS = 4

TARGET_LAYER_GETTER = {
    "mobilenet_v3_large": lambda m: [m.features[-1]],
    "efficientnet_b0": lambda m: [m.features[-1]],
    "xception": lambda m: [m.act4],
}


def pick_fixed_sample():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    real = [r for r in rows if r["label"] == "0"]
    fake = [r for r in rows if r["label"] == "1"]
    rng = random.Random(SEED)
    sample = rng.sample(real, N_PER_CLASS) + rng.sample(fake, N_PER_CLASS)
    return sample


def load_image_tensor(archive_path, transform):
    img = Image.open(IMAGES_DIR / archive_path).convert("RGB")
    tensor = transform(img)
    resized_rgb = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    return tensor, resized_rgb


def find_misclassified(model, device, test_dataset, max_n=2, batch_size=64, num_workers=4):
    """Scan the full test split once, in batches, to find a couple of false
    positives and false negatives for this specific model, for the
    explainability figures. Reuses common.ManifestDataset rather than
    re-parsing the manifest and running one image at a time."""
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    false_positives, false_negatives = [], []
    offset = 0
    with torch.no_grad():
        for images, labels, _sources in loader:
            if len(false_positives) >= max_n and len(false_negatives) >= max_n:
                break
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images).squeeze(1)).cpu()
            preds = (probs >= 0.5).float()
            for i in range(len(labels)):
                true_label = int(labels[i].item())
                pred = int(preds[i].item())
                row = test_dataset.rows[offset + i]
                if pred == 1 and true_label == 0 and len(false_positives) < max_n:
                    false_positives.append(row)
                elif pred == 0 and true_label == 1 and len(false_negatives) < max_n:
                    false_negatives.append(row)
            offset += len(labels)
    return false_positives + false_negatives


def run_for_model(model_name, run_name, sample, device):
    model = build_model(model_name).to(device)
    checkpoint = ROOT / "runs" / run_name / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    target_layers = TARGET_LAYER_GETTER[model_name](model)
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = eval_transform()
    out_dir = OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = ManifestDataset(MANIFEST, "test", eval_transform())
    misclassified = find_misclassified(model, device, test_dataset)
    print(f"[{model_name}] found {len(misclassified)} misclassified examples for explainability figures")
    full_sample = sample + misclassified

    results = []
    for row in full_sample:
        tensor, rgb = load_image_tensor(row["archive_path"], transform)
        input_tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(input_tensor).squeeze(1)
            prob = torch.sigmoid(logit).item()
        pred = 1 if prob >= 0.5 else 0
        true_label = int(row["label"])

        # With a single-logit binary head, an unspecified target defaults to
        # index 0 of the output, which always maximises the "fake" logit
        # regardless of what the model actually predicted. That misexplains
        # every correctly-classified real image and every false negative
        # (their maps would show what pushes toward "fake", not what
        # supports the model's actual "real" call). BinaryClassifierOutputTarget
        # negates the logit for the predicted-real case so the map explains
        # the class the model actually chose.
        target = [BinaryClassifierOutputTarget(pred)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        stem = Path(row["archive_path"]).stem
        out_name = f"{stem}__true{true_label}_pred{pred}_p{prob:.2f}.png"
        Image.fromarray(overlay).save(out_dir / out_name)

        results.append({
            "archive_path": row["archive_path"],
            "source": row["source"],
            "true_label": true_label,
            "predicted_label": pred,
            "predicted_prob": prob,
            "correct": pred == true_label,
            "output_image": str((out_dir / out_name).relative_to(ROOT)),
        })

    with open(out_dir / "gradcam_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_name}] wrote {len(results)} Grad-CAM overlays to {out_dir}")
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = pick_fixed_sample()
    print("fixed sample:", [r["archive_path"] for r in sample])

    all_results = {}
    for model_name, run_name in [
        ("mobilenet_v3_large", "mobilenet_v3_large_seed0"),
        ("efficientnet_b0", "efficientnet_b0_seed0"),
        ("xception", "xception_seed0"),
    ]:
        all_results[model_name] = run_for_model(model_name, run_name, sample, device)

    with open(OUT_DIR / "gradcam_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
