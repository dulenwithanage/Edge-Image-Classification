import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import time

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # save figures to files; no pop-up window
import matplotlib.pyplot as plt

from data import get_datasets
from modelA import build_model_a
from modelB import build_model_b

EPOCHS = 20
SEED = 42


def get_optimizer(name):
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=1e-3)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=1e-2)
    if name == "sgd_momentum":
        return tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


class EpochTimer(tf.keras.callbacks.Callback):
    """Records wall-clock seconds for every epoch (needed for the comparison table)."""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self._t0)


def plot_history(history, tag):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, h["loss"], "b-o", markersize=3, label="Training loss")
    ax[0].plot(epochs, h["val_loss"], "r-o", markersize=3, label="Validation loss")
    ax[0].set(title=f"{tag}: loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax[0].legend()

    ax[1].plot(epochs, h["accuracy"], "b-o", markersize=3, label="Training acc")
    ax[1].plot(epochs, h["val_accuracy"], "r-o", markersize=3, label="Validation acc")
    ax[1].set(title=f"{tag}: accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax[1].legend()

    plt.tight_layout()
    path = os.path.join("plots", f"{tag}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    # Usage: python src/train.py <A|B> <adam|sgd|sgd_momentum>
    model_name = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    opt_name = sys.argv[2].lower() if len(sys.argv) > 2 else "adam"
    tag = f"{model_name}_{opt_name}"

    tf.keras.utils.set_random_seed(SEED)

    train, val, _ = get_datasets()
    model = build_model_a() if model_name == "A" else build_model_b()

    model.compile(
        optimizer=get_optimizer(opt_name),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    timer = EpochTimer()
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("models", f"{tag}.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    history = model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS,
        callbacks=[timer, ckpt],
        verbose=2,
    )

    # Save the numbers so the report tables can be built later
    out = {
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "epoch_times": timer.times,
        "trainable_params": int(sum(int(tf.size(w)) for w in model.trainable_weights)),
    }
    with open(os.path.join("results", f"{tag}_history.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    path = plot_history(history, tag)
    print(f"\nSaved curves to {path}")
    print(f"Mean epoch time (epochs 2-{EPOCHS}): {sum(timer.times[1:]) / (len(timer.times) - 1):.1f} s")


if __name__ == "__main__":
    main()