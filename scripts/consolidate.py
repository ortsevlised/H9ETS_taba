import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

SEEDS = [0, 1, 2]
MAIN_RUN_TEMPLATE = {
    "mobilenet_v3_large": "mobilenet_v3_large_seed{seed}",
    "efficientnet_b0": "efficientnet_b0_seed{seed}",
    "xception": "xception_seed{seed}",
}
HOLDOUT_RUNS = {
    "mobilenet_v3_large": "mobilenet_v3_large_holdout_seed0",
    "efficientnet_b0": "efficientnet_b0_holdout_seed0",
    "xception": "xception_holdout_seed0",
}

F1_THRESHOLD_PP = 2.0
RECALL_THRESHOLD_PP = 3.0

# Illustrative deployment prevalence for the PPV discussion (Section: report
# claim that balanced-test precision will not transfer directly to a
# moderation setting where fakes are rare).
ILLUSTRATIVE_PREVALENCE = 0.01


def load(run_name, filename):
    path = RUNS_DIR / run_name / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_codecarbon_energy_kwh(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["energy_consumed"])


def mean_std(values):
    """Sample standard deviation (ddof=1), not population stdev: with only
    3 seeds, statistics.pstdev understates the sample estimate by ~18%
    (pstdev = stdev * sqrt((n-1)/n) = stdev * sqrt(2/3) for n=3)."""
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_per_source(seed_per_source_dicts):
    """Each seed's per_source dict maps source -> {n, recall} or {n, false_positive_rate}.
    Aggregate the rate (not n, which is identical across seeds by construction
    of the manifest) as mean +/- sample std across seeds."""
    by_source = defaultdict(lambda: {"recall": [], "false_positive_rate": [], "n": None})
    for per_source in seed_per_source_dicts:
        for source, stats in per_source.items():
            by_source[source]["n"] = stats["n"]
            if "recall" in stats:
                by_source[source]["recall"].append(stats["recall"])
            if "false_positive_rate" in stats:
                by_source[source]["false_positive_rate"].append(stats["false_positive_rate"])

    result = {}
    for source, vals in by_source.items():
        entry = {"n": vals["n"]}
        if vals["recall"]:
            mean, std = mean_std(vals["recall"])
            entry["recall_mean"], entry["recall_std"] = mean, std
        if vals["false_positive_rate"]:
            mean, std = mean_std(vals["false_positive_rate"])
            entry["false_positive_rate_mean"], entry["false_positive_rate_std"] = mean, std
        result[source] = entry
    return result


def ppv_at_prevalence(recall, fpr, prevalence):
    """Positive predictive value implied by a given fake-image prevalence,
    holding the model's recall (true positive rate) and false-positive rate
    fixed: PPV = (prevalence * recall) / (prevalence * recall + (1 - prevalence) * fpr)."""
    numerator = prevalence * recall
    denominator = numerator + (1 - prevalence) * fpr
    return numerator / denominator if denominator > 0 else None


def main():
    per_seed_rows = []
    model_aggregates = {}

    for model, template in MAIN_RUN_TEMPLATE.items():
        seed_records = []
        for seed in SEEDS:
            run_name = template.format(seed=seed)
            train = load(run_name, "train_result.json")
            test = load(run_name, "eval_test.json")
            dm = test["detection_metrics"]
            ib = test["inference_benchmark"]
            cm = dm["confusion_matrix"]

            train_energy_kwh = read_codecarbon_energy_kwh(
                RUNS_DIR / run_name / "codecarbon_train.csv"
            )

            record = {
                "model": model,
                "run_name": run_name,
                "seed": seed,
                "test_f1": dm["f1"],
                "test_precision": dm["precision"],
                "test_recall": dm["recall"],
                "test_roc_auc": dm["roc_auc"],
                "false_positive_rate": dm["false_positive_rate"],
                "false_negative_rate": dm["false_negative_rate"],
                "confusion_matrix": cm,
                "per_source": dm["per_source"],
                "num_params": train["num_params"],
                "model_checkpoint_size_mb": train["model_checkpoint_size_mb"],
                "training_duration_s": train["training_duration_s"],
                "peak_gpu_memory_mb": train["peak_gpu_memory_mb"],
                "training_emissions_kg_co2e": train["emissions_kg_co2e"],
                "training_energy_kwh": train_energy_kwh,
                "epochs_run": train["epochs_run"],
                "best_epoch": train["best_epoch"],
                "inference_latency_ms_per_image": ib["latency_ms_per_image"],
                "inference_emissions_kg_co2e_per_1000": ib["emissions_kg_co2e_per_1000_images"],
                "inference_energy_kwh_per_1000": ib["energy_kwh_per_1000_images"],
                "inference_total_images_processed": ib["total_images_processed"],
                "inference_actual_duration_s": ib["actual_duration_s"],
            }
            per_seed_rows.append(record)
            seed_records.append(record)

        f1_mean, f1_std = mean_std([r["test_f1"] for r in seed_records])
        recall_mean, recall_std = mean_std([r["test_recall"] for r in seed_records])
        precision_mean, precision_std = mean_std([r["test_precision"] for r in seed_records])
        roc_auc_mean, roc_auc_std = mean_std([r["test_roc_auc"] for r in seed_records])
        fpr_mean, fpr_std = mean_std([r["false_positive_rate"] for r in seed_records])
        train_co2e_mean, train_co2e_std = mean_std([r["training_emissions_kg_co2e"] for r in seed_records])
        train_kwh_mean, train_kwh_std = mean_std([r["training_energy_kwh"] for r in seed_records])
        infer_co2e_mean, infer_co2e_std = mean_std(
            [r["inference_emissions_kg_co2e_per_1000"] for r in seed_records]
        )
        infer_kwh_mean, infer_kwh_std = mean_std(
            [r["inference_energy_kwh_per_1000"] for r in seed_records]
        )
        infer_latency_mean, infer_latency_std = mean_std(
            [r["inference_latency_ms_per_image"] for r in seed_records]
        )
        peak_gpu_mem_mean, peak_gpu_mem_std = mean_std(
            [r["peak_gpu_memory_mb"] for r in seed_records]
        )

        cm_keys = ["tn", "fp", "fn", "tp"]
        cm_mean = {k: statistics.fmean([r["confusion_matrix"][k] for r in seed_records]) for k in cm_keys}

        per_source_agg = aggregate_per_source([r["per_source"] for r in seed_records])

        ppv_illustrative = ppv_at_prevalence(recall_mean, fpr_mean, ILLUSTRATIVE_PREVALENCE)

        model_aggregates[model] = {
            "model": model,
            "n_seeds": len(seed_records),
            "test_f1_mean": f1_mean, "test_f1_std": f1_std,
            "test_recall_mean": recall_mean, "test_recall_std": recall_std,
            "test_precision_mean": precision_mean, "test_precision_std": precision_std,
            "test_roc_auc_mean": roc_auc_mean, "test_roc_auc_std": roc_auc_std,
            "false_positive_rate_mean": fpr_mean, "false_positive_rate_std": fpr_std,
            "confusion_matrix_mean": cm_mean,
            "per_source": per_source_agg,
            "training_emissions_kg_co2e_mean": train_co2e_mean,
            "training_emissions_kg_co2e_std": train_co2e_std,
            "training_energy_kwh_mean": train_kwh_mean,
            "training_energy_kwh_std": train_kwh_std,
            "inference_emissions_kg_co2e_per_1000_mean": infer_co2e_mean,
            "inference_emissions_kg_co2e_per_1000_std": infer_co2e_std,
            "inference_energy_kwh_per_1000_mean": infer_kwh_mean,
            "inference_energy_kwh_per_1000_std": infer_kwh_std,
            "inference_latency_ms_per_image_mean": infer_latency_mean,
            "inference_latency_ms_per_image_std": infer_latency_std,
            "peak_gpu_memory_mb_mean": peak_gpu_mem_mean,
            "peak_gpu_memory_mb_std": peak_gpu_mem_std,
            "num_params": seed_records[0]["num_params"],
            "model_checkpoint_size_mb": seed_records[0]["model_checkpoint_size_mb"],
            "ppv_at_illustrative_prevalence": ppv_illustrative,
            "illustrative_prevalence": ILLUSTRATIVE_PREVALENCE,
        }

    best_f1 = max(a["test_f1_mean"] for a in model_aggregates.values())
    best_recall = max(a["test_recall_mean"] for a in model_aggregates.values())
    for a in model_aggregates.values():
        f1_gap_pp = (best_f1 - a["test_f1_mean"]) * 100
        recall_gap_pp = (best_recall - a["test_recall_mean"]) * 100
        a["f1_gap_pp_from_best"] = f1_gap_pp
        a["recall_gap_pp_from_best"] = recall_gap_pp
        a["viable_by_decision_rule"] = (
            f1_gap_pp <= F1_THRESHOLD_PP and recall_gap_pp <= RECALL_THRESHOLD_PP
        )

    viable = [a for a in model_aggregates.values() if a["viable_by_decision_rule"]]
    recommended = (
        min(viable, key=lambda a: a["inference_emissions_kg_co2e_per_1000_mean"])
        if viable else None
    )

    generalisation_rows = []
    for model, run_name in HOLDOUT_RUNS.items():
        gen = load(run_name, "eval_generalisation.json")
        dm = gen["detection_metrics"]
        generalisation_rows.append({
            "model": model,
            "run_name": run_name,
            "generalisation_f1": dm["f1"],
            "generalisation_recall": dm["recall"],
            "generalisation_precision": dm["precision"],
            "generalisation_roc_auc": dm["roc_auc"],
        })

    idle_baseline = None
    idle_path = RUNS_DIR / "idle_baseline.json"
    if idle_path.exists():
        idle_baseline = json.load(open(idle_path, encoding="utf-8"))

    summary = {
        "decision_rule": {
            "f1_threshold_pp": F1_THRESHOLD_PP,
            "recall_threshold_pp": RECALL_THRESHOLD_PP,
            "best_f1_mean": best_f1,
            "best_recall_mean": best_recall,
        },
        "per_seed_results": per_seed_rows,
        "model_aggregates": list(model_aggregates.values()),
        "viable_models": [a["model"] for a in viable],
        "recommended_model": recommended["model"] if recommended else None,
        "generalisation_results": generalisation_rows,
        "idle_baseline": idle_baseline,
    }

    out_path = RUNS_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Main comparison (test split, mean +/- SAMPLE std over 3 seeds) ===")
    for a in model_aggregates.values():
        print(f"{a['model']:20s} "
              f"F1={a['test_f1_mean']:.4f}+/-{a['test_f1_std']:.4f} (gap {a['f1_gap_pp_from_best']:.2f}pp) "
              f"recall={a['test_recall_mean']:.4f}+/-{a['test_recall_std']:.4f} (gap {a['recall_gap_pp_from_best']:.2f}pp) "
              f"FPR={a['false_positive_rate_mean']:.4f}+/-{a['false_positive_rate_std']:.4f} "
              f"viable={a['viable_by_decision_rule']} "
              f"params={a['num_params']/1e6:.2f}M")
        print(f"    train: {a['training_emissions_kg_co2e_mean']*1000:.3f}+/-{a['training_emissions_kg_co2e_std']*1000:.3f} g CO2e, "
              f"{a['training_energy_kwh_mean']*1000:.3f}+/-{a['training_energy_kwh_std']*1000:.3f} Wh")
        print(f"    infer/1000img: {a['inference_emissions_kg_co2e_per_1000_mean']*1e6:.3f}+/-{a['inference_emissions_kg_co2e_per_1000_std']*1e6:.3f} mg CO2e, "
              f"{a['inference_energy_kwh_per_1000_mean']*1e6:.3f}+/-{a['inference_energy_kwh_per_1000_std']*1e6:.3f} mWh, "
              f"{a['inference_latency_ms_per_image_mean']:.4f}+/-{a['inference_latency_ms_per_image_std']:.4f} ms/img")
        print(f"    PPV at {a['illustrative_prevalence']*100:.0f}% fake prevalence: {a['ppv_at_illustrative_prevalence']*100:.1f}%")
    print()
    print("Viable models:", [a["model"] for a in viable])
    print("Recommended model (lowest mean inference emissions per 1000 images among viable):",
          summary["recommended_model"])
    print()
    print("=== Generalisation check (unseen StyleGAN-FFHQ, seed 0 only) ===")
    for r in generalisation_rows:
        print(f"{r['model']:20s} F1={r['generalisation_f1']:.4f} recall={r['generalisation_recall']:.4f} "
              f"precision={r['generalisation_precision']:.4f} roc_auc={r['generalisation_roc_auc']:.4f}")
    print()
    if idle_baseline:
        print(f"Idle baseline: {idle_baseline['energy_kwh']*1000:.4f} Wh over {idle_baseline['duration_s']:.1f}s "
              f"({idle_baseline['emissions_kg_co2e']*1e6:.2f} mg CO2e)")
    print(f"summary written to {out_path}")


if __name__ == "__main__":
    main()
