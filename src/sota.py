import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")  # save figures to files; no pop-up window
import matplotlib.pyplot as plt

from data import get_datasets, IMG_SIZE, NUM_CLASSES
from train import EpochTimer

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
SEED = 42
HEAD_EPOCHS = 5    # phase 1: backbone frozen, train only the new classifier head
FINE_EPOCHS = 15   # phase 2: unfreeze everything, small learning rate


def build(name):
    """Pretrained backbone + small classifier head.

    The data pipeline gives pixels in [0, 1]. Each pretrained network expects its
    own input range, so a Rescaling layer inside the model converts it:
      MobileNetV2    expects [-1, 1]  ->  2x - 1
      EfficientNetB0 expects [0, 255] ->  255x  (it normalises internally)
    """
    shape = (*IMG_SIZE, 3)
    inp = layers.Input(shape=shape)

    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=shape)
        x = layers.Rescaling(2.0, offset=-1.0)(inp)
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=shape)
        x = layers.Rescaling(255.0)(inp)
    else:
        raise ValueError("name must be 'mobilenetv2' or 'efficientnetb0'")

    base.trainable = False  # phase 1: frozen (batch-norm layers stay in inference mode)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inp, out, name=name), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_ckpt(path):
    return tf.keras.callbacks.ModelCheckpoint(
        path, monitor="val_loss", save_best_only=True, save_weights_only=True)


def plot_curves(hist, tag):
    epochs = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, (key, ylabel) in zip(ax, [("loss", "Cross-entropy loss"), ("accuracy", "Accuracy")]):
        a.plot(epochs, hist[key], "b-o", markersize=3, label=f"Training {key}")
        a.plot(epochs, hist["val_" + key], "r-o", markersize=3, label=f"Validation {key}")
        a.axvline(HEAD_EPOCHS + 0.5, color="gray", linestyle="--", label="Backbone unfrozen")
        a.set(title=f"{tag}: {key}", xlabel="Epoch", ylabel=ylabel)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{tag}_curves.png"), dpi=150)
    plt.close(fig)


def plot_confusion(cm, tag):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{tag}: confusion matrix (test)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{tag}_confusion.png"), dpi=150)
    plt.close(fig)


def main():
    # Usage: python src/sota.py <mobilenetv2|efficientnetb0>
    name = sys.argv[1].lower() if len(sys.argv) > 1 else "mobilenetv2"
    tag = f"{name}_finetuned"
    tf.keras.utils.set_random_seed(SEED)

    train, val, test = get_datasets()   # same 70/15/15 split as Models A and B
    model, base = build(name)
    ckpt_path = os.path.join("models", f"{tag}.weights.h5")

    # Phase 1: train only the new head
    compile_model(model, 1e-3)
    t1 = EpochTimer()
    h1 = model.fit(train, validation_data=val, epochs=HEAD_EPOCHS,
                   callbacks=[t1, make_ckpt(ckpt_path)], verbose=2)

    # Phase 2: unfreeze the whole network and fine-tune with a small learning rate
    base.trainable = True
    compile_model(model, 1e-4)   # must recompile after changing trainable
    t2 = EpochTimer()
    h2 = model.fit(train, validation_data=val, epochs=FINE_EPOCHS,
                   callbacks=[t2, make_ckpt(ckpt_path)], verbose=2)

    model.load_weights(ckpt_path)   # best validation-loss weights

    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    plot_curves(hist, tag)

    # ---- Test-set evaluation ----
    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = model.predict(test, verbose=0).argmax(axis=1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    acc = float(tp.sum() / cm.sum())
    plot_confusion(cm, tag)

    # ---- Resource numbers ----
    total_params = int(model.count_params())
    trainable_params = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    size_mb = os.path.getsize(ckpt_path) / 1024 ** 2   # weights-only file, no optimizer state
    est_mb = total_params * 4 / 1024 ** 2               # float32 estimate
    epoch_time = float(np.mean(t2.times))               # full fine-tuning epochs (phase 2)

    metrics = {
        "model": name,
        "test_accuracy": acc,
        "precision_per_class": dict(zip(CLASSES, map(float, precision))),
        "recall_per_class": dict(zip(CLASSES, map(float, recall))),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion_matrix": cm.tolist(),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb_weights_file": size_mb,
        "size_mb_float32_estimate": est_mb,
        "mean_epoch_time_s_phase2": epoch_time,
        "history": {k: [float(v) for v in vs] for k, vs in hist.items()},
    }
    with open(os.path.join("results", f"{tag}_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n=== {name} (fine-tuned) ===")
    print(f"Test accuracy     : {acc:.4f}")
    print(f"Macro precision   : {precision.mean():.4f}")
    print(f"Macro recall      : {recall.mean():.4f}")
    print(f"Total params      : {total_params:,}  (trainable {trainable_params:,})")
    print(f"Model size        : {size_mb:.2f} MB  (float32 estimate {est_mb:.2f} MB)")
    print(f"Mean epoch time   : {epoch_time:.1f} s")
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()