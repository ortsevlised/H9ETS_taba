import argparse
import json
import time
from pathlib import Path

import torch
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from common import ManifestDataset, build_model, eval_transform, set_seed, train_transform

ROOT = Path(__file__).resolve().parent.parent


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0
    all_labels, all_preds = [], []
    with torch.set_grad_enabled(is_train):
        for images, labels, _sources in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / n_samples
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return {"loss": avg_loss, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3_large", "efficientnet_b0", "xception"])
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "main_manifest.csv"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ManifestDataset(args.manifest, args.train_split, train_transform())
    val_ds = ManifestDataset(args.manifest, args.val_split, eval_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tracker = OfflineEmissionsTracker(
        country_iso_code="IRL",
        project_name=f"taba_train_{args.run_name}",
        output_dir=str(run_dir),
        output_file="codecarbon_train.csv",
        log_level="error",
        measure_power_secs=5,
    )
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    tracker.start()
    start_time = time.perf_counter()

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"[{args.run_name}] epoch {epoch}: train_f1={train_metrics['f1']:.4f} "
              f"val_f1={val_metrics['f1']:.4f} val_loss={val_metrics['loss']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[{args.run_name}] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    training_duration_s = time.perf_counter() - start_time
    emissions_kg = tracker.stop()

    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    model_size_mb = (run_dir / "best.pt").stat().st_size / (1024 ** 2)

    result = {
        "model": args.model,
        "run_name": args.run_name,
        "manifest": args.manifest,
        "seed": args.seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "history": history,
        "num_params": num_params,
        "trainable_params": trainable_params,
        "model_checkpoint_size_mb": model_size_mb,
        "training_duration_s": training_duration_s,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "emissions_kg_co2e": emissions_kg,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(run_dir / "train_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.run_name}] done. best_val_f1={best_f1:.4f} duration={training_duration_s:.1f}s "
          f"emissions={emissions_kg}")


if __name__ == "__main__":
    main()
