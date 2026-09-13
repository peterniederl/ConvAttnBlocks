# ResNet Attention Experiments on Tiny ImageNet

This repository provides a reproducible training pipeline for comparing a small ResNet-style image classifier with three attention configurations:

- **Baseline**: no attention block
- **SE**: squeeze-and-excitation channel attention
- **CBAM**: channel and spatial attention

The experiments use the Tiny ImageNet dataset with a shared data pipeline, augmentation settings, optimizer, loss, learning-rate schedule, and evaluation procedure. This keeps comparisons between model variants consistent.

## Dataset

Download and extract Tiny ImageNet so the repository contains:

```text
tiny-imagenet-200/
├── train/
├── val/
├── wnids.txt
└── words.txt
```

The dataset contains 200 classes with 64 x 64 RGB images. The local dataset directory is ignored by Git because it is input data rather than source code.

## Running experiments

The recommended entry point runs three repetitions for each model by default:

```bash
python3 run_experiments.py
```

This runs the baseline, SE, and CBAM models for three runs each. To choose the number of repetitions or models:

```bash
python3 run_experiments.py --runs 5
python3 run_experiments.py --runs 3 --models baseline se cbam
python3 run_experiments.py --runs 1 --models cbam
```

Each model is trained for 100 epochs by default. A new model is created for each run, and the previous TensorFlow session is cleared before the next model is built.

Individual model entry points are also available:

```bash
python3 run_baseline.py
python3 run_se.py
python3 run_cbam.py
```

## Results

Each completed run is saved under `results/<run-name>/`, including:

- experiment and environment settings
- model configuration and summary
- epoch-by-epoch training logs
- training history and final evaluation metrics
- best and final Keras model files

The multi-run script also creates:

```text
results/comparison.json
results/comparison.csv
```

These files contain the metrics for every model and repetition, including best validation accuracy, final validation accuracy, and evaluation accuracy.

To plot completed runs:

```bash
python3 visualize_results.py
python3 visualize_results.py baseline_run_01 se_run_01 cbam_run_01 \
    --output results/comparison.png
```

## Project structure

```text
experiment_config.py   Shared training configuration
data_pipeline.py       Tiny ImageNet loading and augmentation
model_factory.py       ResNet, SE, and CBAM model construction
experiment_runner.py   Shared training and evaluation logic
run_experiments.py     Repeated multi-model experiment runner
run_baseline.py        Baseline convenience entry point
run_se.py              SE convenience entry point
run_cbam.py            CBAM convenience entry point
visualize_results.py   Result loading and plotting
callbacks.py           Learning-rate scheduling callback
```

The older ConvMixer notebook and its supporting utilities are retained separately as historical research material and are not part of the current ResNet experiment pipeline.
