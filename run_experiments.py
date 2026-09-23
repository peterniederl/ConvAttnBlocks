import argparse
import csv
import gc
import json
from pathlib import Path

import tensorflow as tf

from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment


MODEL_ATTENTION = {
    "baseline": "none",
    "se": "se",
    "cbam": "cbam",
    "axial_multiply": "axial_multiply",
    "axial_multiply_pointwise": "axial_multiply_pointwise",
    "axial_sum": "axial_sum",
    "axial_sum_tanh": "axial_sum_residual_gate",
    "axial_fused": "axial_sum_weighted",
    "axial_sum_pointwise": "axial_sum_pointwise",
    "axial_multiply_postpointwise": "axial_multiply_postpointwise",
    "axial_sum_postpointwise": "axial_sum_postpointwise",
    "axial_sum_se": "axial_sum_se",
    "se_axial_sum": "se_axial_sum",
    "depthwise_gate_upsample_stride2": "depthwise_gate_upsample_stride2",
    "depthwise_gate_upsample_stride4": "depthwise_gate_upsample_stride4",
    "projected_gate_upsample_stride2": "projected_gate_upsample_stride2",
    "projected_gate_upsample_stride4": "projected_gate_upsample_stride4",
    "global_projected_gate_upsample_stride2": "global_projected_gate_upsample_stride2",
    "global_projected_gate_upsample_stride4": "global_projected_gate_upsample_stride4",
    "axial_conv_attention": "axial_conv_attention",
    "axial_conv_attention_projected": "axial_conv_attention_projected",
    "axial_conv_attention_query_multiply": "axial_conv_attention_query_multiply",
    "axial_conv_self_attention": "axial_conv_self_attention",
    "axial_conv_self_attention_no_projection": "axial_conv_self_attention_no_projection",
    "axial_avg_pool_dual_norm": "axial_avg_pool_dual_norm",
    "axial_full_conv_gate": "axial_full_conv_gate",
    "multi_stage_spatial_mixer": "multi_stage_spatial_mixer",
}


class ConfiguredExperiment(ResNetExperiment):
    def __init__(self, run_name, attention, seed, results_dir):
        self.run_name = run_name
        self.attention = attention
        self.seed = seed
        self.results_dir = results_dir

    def config(self):
        return ExperimentConfig(
            run_name=self.run_name,
            attention=self.attention,
            seed=self.seed,
            results_dir=self.results_dir,
        )


def _best_validation_metrics(history):
    val_accuracy = history.get("val_accuracy", [])
    val_loss = history.get("val_loss", [])
    if not val_accuracy:
        return None, None
    best_index = max(range(len(val_accuracy)), key=val_accuracy.__getitem__)
    best_loss = val_loss[best_index] if val_loss else None
    return val_accuracy[best_index], best_loss


def _read_result(results_dir, model_name, run_number, seed):
    run_name = f"{model_name}_run_{run_number:02d}"
    run_dir = Path(results_dir) / run_name
    history = json.loads((run_dir / "history.json").read_text())
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    best_val_accuracy, best_val_loss = _best_validation_metrics(history)
    return {
        "model": model_name,
        "run": run_number,
        "seed": seed,
        "run_name": run_name,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_val_loss": history["val_loss"][-1],
        "evaluation_accuracy": evaluation["accuracy"],
        "evaluation_loss": evaluation["loss"],
    }


def _average_best_validation_accuracy(results_dir):
    scores = []
    results_path = Path(results_dir)
    for history_path in results_path.glob("*_run_*/history.json"):
        if not _run_is_complete(history_path.parent):
            continue
        history = json.loads(history_path.read_text())
        val_accuracy = history.get("val_accuracy") or []
        if val_accuracy:
            scores.append(max(val_accuracy))
    return sum(scores) / len(scores) if scores else None


def _run_is_complete(run_dir):
    return all(
        (run_dir / artifact).is_file()
        for artifact in ("history.json", "evaluation.json", "final_model.keras")
    )


def _next_run_number(results_dir, model_name):
    run_number = 1
    while True:
        run_dir = Path(results_dir) / f"{model_name}_run_{run_number:02d}"
        if not run_dir.exists() or not _run_is_complete(run_dir):
            return run_number
        run_number += 1


def write_summary(results_dir, records):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "comparison.json").write_text(json.dumps(records, indent=2))

    fields = list(records[0]) if records else []
    with (results_path / "comparison.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_experiments(run_counts, model_names, base_seed, results_dir):
    records = []
    for model_name in model_names:
        for _ in range(run_counts[model_name]):
            run_number = _next_run_number(results_dir, model_name)
            seed = base_seed + run_number - 1
            run_name = f"{model_name}_run_{run_number:02d}"
            print(f"\nStarting {run_name} with seed {seed}")
            experiment = ConfiguredExperiment(
                run_name=run_name,
                attention=MODEL_ATTENTION[model_name],
                seed=seed,
                results_dir=results_dir,
            )
            experiment.run()
            records.append(_read_result(results_dir, model_name, run_number, seed))
            tf.keras.backend.clear_session()
            gc.collect()
            write_summary(results_dir, records)
            experiment.publish_results()

            average_best_accuracy = _average_best_validation_accuracy(results_dir)
            current_best_accuracy = records[-1]["best_val_accuracy"]
            if (
                average_best_accuracy is not None
                and current_best_accuracy < average_best_accuracy
            ):
                print(
                    f"Skipping remaining {model_name} runs: "
                    f"best validation accuracy {current_best_accuracy:.4f} is below "
                    f"the global average {average_best_accuracy:.4f}."
                )
                break

    return records


def _parse_run_counts(run_specs, model_names):
    run_counts = {model_name: 3 for model_name in model_names}
    for run_spec in run_specs or []:
        try:
            model_name, count_text = run_spec.split("=", 1)
            count = int(count_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid run specification {run_spec!r}; use MODEL=COUNT"
            ) from error
        if model_name not in MODEL_ATTENTION:
            raise ValueError(f"Unknown model in --runs: {model_name}")
        if model_name not in model_names:
            raise ValueError(
                f"Model {model_name!r} must also be selected with --models"
            )
        if count < 1:
            raise ValueError(f"Run count for {model_name} must be at least 1")
        run_counts[model_name] = count
    return run_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated ResNet comparisons for baseline, SE, CBAM, and axial variants."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="MODEL=COUNT",
        help="Runs per model, for example baseline=2 axial_sum=5 (default: 3 each).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_ATTENTION),
        default=list(MODEL_ATTENTION),
        help="Models to compare (default: all registered models).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the first run.")
    parser.add_argument("--results-dir", default="results", help="Directory for run outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_counts = _parse_run_counts(args.runs, args.models)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    records = run_experiments(run_counts, args.models, args.seed, args.results_dir)
    print(f"\nCompleted {len(records)} runs.")
    print(f"Comparison written to {Path(args.results_dir) / 'comparison.csv'}")


if __name__ == "__main__":
    main()