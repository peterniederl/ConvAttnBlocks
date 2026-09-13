import json
import platform
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from callbacks import CosineAnnealingScheduler
from data_pipeline import TinyImageNetData
from model_factory import build_resnet


class ResNetExperiment:
    """Shared runner. A new experiment only needs to override config()."""

    def config(self):
        raise NotImplementedError

    def build_model(self, config):
        return build_resnet(config)

    def run(self):
        config = self.config()
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        run_dir = Path(config.results_dir) / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "settings.json", config.to_dict())
        self._write_json(run_dir / "environment.json", {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
        })

        train_ds, val_ds = TinyImageNetData(config).datasets()
        model = self.build_model(config)
        optimizer = tfa.optimizers.AdamW(learning_rate=config.base_lr, weight_decay=config.weight_decay, beta_1=config.beta_1, beta_2=config.beta_2)
        model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=config.label_smoothing), metrics=["accuracy"])
        self._write_json(run_dir / "model_config.json", json.loads(model.to_json()))
        with (run_dir / "model_summary.txt").open("w") as summary_file:
            with redirect_stdout(summary_file):
                model.summary()

        callbacks = [
            CosineAnnealingScheduler(config.base_lr, config.min_lr, config.epochs, config.warmup_epochs),
            keras.callbacks.CSVLogger(run_dir / "training_log.csv"),
            keras.callbacks.ModelCheckpoint(run_dir / "best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True),
        ]
        history = model.fit(train_ds, epochs=config.epochs, validation_data=val_ds, callbacks=callbacks)
        evaluation = model.evaluate(val_ds, return_dict=True)
        self._write_json(run_dir / "history.json", history.history)
        self._write_json(run_dir / "evaluation.json", {key: float(value) for key, value in evaluation.items()})
        model.save(run_dir / "final_model.keras")
        return run_dir

    @staticmethod
    def _write_json(path, value):
        with path.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
