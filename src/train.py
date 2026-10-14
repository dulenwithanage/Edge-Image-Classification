import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
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

SEED = 42
MIN_EPOCHS = 20   # assignment minimum; EarlyStopping never stops before this

# ---------------------------------------------------------------------------
# Settings for every (model, optimizer) run. Edit any value here.
#   lr:     starting learning rate. ReduceLROnPlateau lowers it during training;
#           the rate used in each epoch is printed and saved in
#           results/<model>_<optimizer>_history.json.
#   epochs: number of epochs (keep it >= MIN_EPOCHS for the assignment).
# A third command-line argument overrides epochs for one run:
#   python src/train.py A adam 40
# ---------------------------------------------------------------------------
RUN_CONFIG = {
    ("A", "adam"):         {"lr": 1e-3, "epochs": 20},
    ("A", "sgd"):          {"lr": 5e-3, "epochs": 20},
    ("A", "sgd_momentum"): {"lr": 1e-2, "epochs": 20},
    ("B", "adam"):         {"lr": 1e-3, "epochs": 20},
    ("B", "sgd"):          {"lr": 1e-2, "epochs": 20},
    ("B", "sgd_momentum"): {"lr": 1e-2, "epochs": 20},
}
MOMENTUM = 0.9   # used by sgd_momentum only


def get_optimizer(name, lr):
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "sgd_momentum":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=MOMENTUM)
    raise ValueError(f"Unknown optimizer: {name}")


class LRLogger(tf.keras.callbacks.Callback):
    """Records and prints the learning rate used in every epoch."""

    def on_train_begin(self, logs=None):
        self.lrs = []

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate.numpy())
        self.lrs.append(lr)
        print(f"Epoch {epoch + 1}: learning rate = {lr:.6g}")


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
    # Usage: python src/train.py <A|B> <adam|sgd|sgd_momentum> [epochs]
    model_name = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    opt_name = sys.argv[2].lower() if len(sys.argv) > 2 else "adam"
    tag = f"{model_name}_{opt_name}"
    if (model_name, opt_name) not in RUN_CONFIG:
        raise ValueError(f"No settings in RUN_CONFIG for model {model_name} with {opt_name}")
    cfg = RUN_CONFIG[(model_name, opt_name)]
    lr = cfg["lr"]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else cfg["epochs"]
    if epochs < MIN_EPOCHS:
        print(f"Note: {epochs} epochs is below the assignment minimum of {MIN_EPOCHS}")
    print(f"Training model {model_name} with {opt_name}: starting lr {lr:g}, {epochs} epochs")

    tf.keras.utils.set_random_seed(SEED)

    train, val, _ = get_datasets()
    model = build_model_a() if model_name == "A" else build_model_b()

    model.compile(
        optimizer=get_optimizer(opt_name, lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    timer = EpochTimer()
    lr_log = LRLogger()
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("models", f"{tag}.keras"),
        monitor="val_loss",
        save_best_only=True,
    )
    # Halve the learning rate when validation loss stops improving for 2 epochs.
    # Smaller late-stage steps stop the validation loss from jumping around.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    # Stop when validation loss has not improved for 5 epochs and roll back to the
    # best weights. start_from_epoch keeps the 20 epochs the assignment requires;
    # it only matters for longer runs (e.g. python src/train.py A adam 40).
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True,
        start_from_epoch=MIN_EPOCHS, verbose=1)

    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        callbacks=[timer, lr_log, ckpt, reduce_lr, early_stop],
        verbose=2,
    )

    # Save the numbers so the report tables can be built later
    out = {
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "epoch_times": timer.times,
        "initial_learning_rate": lr,
        "epochs_requested": epochs,
        "epochs_run": len(timer.times),
        "momentum": MOMENTUM if opt_name == "sgd_momentum" else 0.0,
        "learning_rate_per_epoch": lr_log.lrs,
        "trainable_params": int(sum(int(tf.size(w)) for w in model.trainable_weights)),
    }
    with open(os.path.join("results", f"{tag}_history.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    path = plot_history(history, tag)
    print(f"\nSaved curves to {path}")
    print(f"Learning rate: started at {lr:g}, ended at {lr_log.lrs[-1]:.6g}")
    print(f"Mean epoch time (epochs 2-{len(timer.times)}): {sum(timer.times[1:]) / (len(timer.times) - 1):.1f} s")


if __name__ == "__main__":
    main()