import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from common import ManifestDataset, build_model, eval_transform, read_codecarbon_energy_kwh

ROOT = Path(__file__).resolve().parent.parent
INFERENCE_BENCH_N = 1000
INFERENCE_BATCH_SIZE = 32
INFERENCE_BENCH_TARGET_DURATION_S = 30.0


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_labels, all_probs, all_sources = [], [], []
    for images, labels, sources in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.tolist())
        all_sources.extend(sources)
    return all_labels, all_probs, all_sources


def per_source_breakdown(labels, preds, sources):
    breakdown = {}
    grouped = defaultdict(lambda: {"labels": [], "preds": []})
    for label, pred, source in zip(labels, preds, sources):
        grouped[source]["labels"].append(label)
        grouped[source]["preds"].append(pred)
    for source, d in grouped.items():
        labs, prds = d["labels"], d["preds"]
        n = len(labs)
        if labs[0] == 0:  # real-only source: report false positive rate
            fpr = sum(1 for l, p in zip(labs, prds) if p == 1) / n
            breakdown[source] = {"n": n, "false_positive_rate": fpr}
        else:  # fake source: report recall (detection rate)
            recall = sum(1 for l, p in zip(labs, prds) if p == 1) / n
            breakdown[source] = {"n": n, "recall": recall}
    return breakdown


@torch.no_grad()
def benchmark_inference(model, dataset, device, run_dir, output_suffix):
    """Model-forward-pass cost only: images are pre-loaded onto the GPU
    before timing starts, so decoding, preprocessing, and host-to-device
    transfer are excluded. A single pass over 1,000 images took ~0.2-0.5s in
    an earlier version of this benchmark, too short for a stable CodeCarbon
    reading; this version loops over the same pre-loaded buffer for at least
    INFERENCE_BENCH_TARGET_DURATION_S, tracking how many full repetitions and
    total images that covers."""
    n = min(INFERENCE_BENCH_N, len(dataset))
    images = torch.stack([dataset[i][0] for i in range(n)]).to(device)

    # warm-up
    for i in range(0, min(64, n), INFERENCE_BATCH_SIZE):
        model(images[i:i + INFERENCE_BATCH_SIZE])
    if device.type == "cuda":
        torch.cuda.synchronize()

    csv_path = run_dir / f"codecarbon_infer_{output_suffix}.csv"
    tracker = OfflineEmissionsTracker(
        country_iso_code="IRL",
        project_name=f"taba_infer_{run_dir.name}_{output_suffix}",
        output_dir=str(run_dir),
        output_file=csv_path.name,
        log_level="error",
        measure_power_secs=1,
    )
    tracker.start()
    start = time.perf_counter()
    n_images_processed = 0
    n_repetitions = 0
    while time.perf_counter() - start < INFERENCE_BENCH_TARGET_DURATION_S:
        for i in range(0, n, INFERENCE_BATCH_SIZE):
            model(images[i:i + INFERENCE_BATCH_SIZE])
            n_images_processed += images[i:i + INFERENCE_BATCH_SIZE].size(0)
        n_repetitions += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    emissions_kg = tracker.stop()
    energy_kwh = read_codecarbon_energy_kwh(csv_path)

    return {
        "note": "model-forward-pass cost only: excludes image decode, preprocessing, and host-to-device transfer",
        "n_images_in_buffer": n,
        "batch_size": INFERENCE_BATCH_SIZE,
        "target_duration_s": INFERENCE_BENCH_TARGET_DURATION_S,
        "actual_duration_s": elapsed_s,
        "repetitions": n_repetitions,
        "total_images_processed": n_images_processed,
        "latency_ms_per_image": (elapsed_s / n_images_processed) * 1000,
        "emissions_kg_co2e_total": emissions_kg,
        "energy_kwh_total": energy_kwh,
        "emissions_kg_co2e_per_1000_images": (emissions_kg / n_images_processed) * 1000,
        "energy_kwh_per_1000_images": (
            (energy_kwh / n_images_processed) * 1000 if energy_kwh is not None else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3_large", "efficientnet_b0", "xception"])
    parser.add_argument("--run-name", required=True, help="matches the run directory under runs/ containing best.pt")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "main_manifest.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-suffix", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / args.run_name
    checkpoint_path = run_dir / "best.pt"

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = ManifestDataset(args.manifest, args.split, eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    labels, probs, sources = collect_predictions(model, loader, device)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    metrics = {
        "n": len(labels),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else None,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else None,
        "per_source": per_source_breakdown(labels, preds, sources),
    }

    inference_bench = benchmark_inference(model, dataset, device, run_dir, args.output_suffix)

    result = {
        "model": args.model,
        "run_name": args.run_name,
        "manifest": args.manifest,
        "split": args.split,
        "detection_metrics": metrics,
        "inference_benchmark": inference_bench,
    }
    out_path = run_dir / f"eval_{args.output_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.run_name}] {args.split}: f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
          f"precision={metrics['precision']:.4f} roc_auc={metrics['roc_auc']}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
