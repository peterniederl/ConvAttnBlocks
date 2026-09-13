import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_runs(results_dir, selected):
    root = Path(results_dir)
    names = selected or sorted(path.name for path in root.iterdir() if path.is_dir())
    runs = {}
    for name in names:
        run_dir = root / name
        history_path = run_dir / "history.json"
        evaluation_path = run_dir / "evaluation.json"
        if history_path.exists() and evaluation_path.exists():
            runs[name] = {
                "history": json.loads(history_path.read_text()),
                "evaluation": json.loads(evaluation_path.read_text()),
            }
    if not runs:
        raise FileNotFoundError("No completed runs with history.json and evaluation.json were found.")
    return runs


def plot_runs(runs, output_path=None):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, run in runs.items():
        history = run["history"]
        epochs = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs, history["accuracy"], label=name)
        axes[0].plot(epochs, history["val_accuracy"], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history["loss"], label=name)
        axes[1].plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
        axes[2].bar(name, run["evaluation"]["accuracy"])
    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    axes[2].set_title("Final evaluation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize one, several, or all ResNet experiment runs.")
    parser.add_argument("runs", nargs="*", help="Run directory names. Omit to include every completed run.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", help="Optional image path instead of opening a plot window.")
    args = parser.parse_args()
    plot_runs(load_runs(args.results_dir, args.runs), args.output)
