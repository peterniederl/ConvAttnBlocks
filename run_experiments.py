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
    "axial": "axial",
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


def write_summary(results_dir, records):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "comparison.json").write_text(json.dumps(records, indent=2))

    fields = list(records[0]) if records else []
    with (results_path / "comparison.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_experiments(number_of_runs, model_names, base_seed, results_dir):
    records = []
    for run_number in range(1, number_of_runs + 1):
        for model_name in model_names:
            seed = base_seed + run_number - 1
            run_name = f"{model_name}_run_{run_number:02d}"
            print(f"\nStarting {run_name} with seed {seed}")
            ConfiguredExperiment(
                run_name=run_name,
                attention=MODEL_ATTENTION[model_name],
                seed=seed,
                results_dir=results_dir,
            ).run()
            records.append(_read_result(results_dir, model_name, run_number, seed))
            tf.keras.backend.clear_session()
            gc.collect()
            write_summary(results_dir, records)

    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated ResNet comparisons for baseline, SE, CBAM, and axial models."
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per model.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_ATTENTION),
        default=list(MODEL_ATTENTION),
        help="Models to compare (default: baseline se cbam axial).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the first run.")
    parser.add_argument("--results-dir", default="results", help="Directory for run outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    records = run_experiments(args.runs, args.models, args.seed, args.results_dir)
    print(f"\nCompleted {len(records)} runs.")
    print(f"Comparison written to {Path(args.results_dir) / 'comparison.csv'}")


if __name__ == "__main__":
    main()