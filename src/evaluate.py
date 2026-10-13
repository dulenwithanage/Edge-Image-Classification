import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data import get_datasets, NUM_CLASSES
from sota import CLASSES, plot_confusion


def count_macs(model):
    """Multiply-accumulate operations for one image (conv, separable conv and dense layers)."""
    macs = 0
    for layer in model.layers:
        if isinstance(layer, layers.SeparableConv2D):
            _, h, w, c_out = layer.output.shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            c_in = layer.input.shape[-1]
            macs += h * w * c_in * k          # depthwise
            macs += h * w * c_in * c_out      # pointwise 1x1
        elif isinstance(layer, layers.Conv2D):
            _, h, w, c_out = layer.output.shape
            k = layer.kernel_size[0] * layer.kernel_size[1]
            macs += h * w * layer.input.shape[-1] * k * c_out
        elif isinstance(layer, layers.Dense):
            macs += layer.input.shape[-1] * layer.units
    return int(macs)


def evaluate(tag, test):
    """Test-set metrics for a model trained by train.py (models/<tag>.keras)."""
    model_path = os.path.join("models", f"{tag}.keras")
    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = model.predict(test, verbose=0).argmax(axis=1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    acc = float(tp.sum() / cm.sum())
    plot_confusion(cm, tag)

    with open(os.path.join("results", f"{tag}_history.json")) as fh:
        hist = json.load(fh)
    times = hist["epoch_times"]
    epoch_time = float(np.mean(times[1:])) if len(times) > 1 else float(times[0])  # skip epoch 1 (cache fill)
    val_loss = hist["history"]["val_loss"]
    best = int(np.argmin(val_loss))

    total_params = int(model.count_params())
    trainable_params = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    metrics = {
        "model": tag,
        "test_accuracy": acc,
        "precision_per_class": dict(zip(CLASSES, map(float, precision))),
        "recall_per_class": dict(zip(CLASSES, map(float, recall))),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion_matrix": cm.tolist(),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs": count_macs(model),
        # .keras file also stores optimizer state; the float32 estimate is the weights alone
        "size_kb_keras_file": os.path.getsize(model_path) / 1024,
        "size_kb_float32_estimate": total_params * 4 / 1024,
        "mean_epoch_time_s": epoch_time,
        "best_epoch": best + 1,
        "best_val_loss": float(val_loss[best]),
        "best_val_accuracy": float(hist["history"]["val_accuracy"][best]),
    }
    with open(os.path.join("results", f"{tag}_test_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"{tag:16s} test acc {acc:.4f}  macro P {precision.mean():.4f}  macro R {recall.mean():.4f}")
    return metrics


def write_tables():
    """Collect every *_test_metrics.json / *_finetuned_metrics.json into results/comparison.md."""
    rows = {}
    for f in sorted(glob.glob(os.path.join("results", "*_test_metrics.json"))):
        with open(f) as fh:
            m = json.load(fh)
        rows[m["model"]] = m
    sota = {}
    for f in sorted(glob.glob(os.path.join("results", "*_finetuned_metrics.json"))):
        with open(f) as fh:
            m = json.load(fh)
        sota[m["model"]] = m

    out = ["## Model A vs Model B (Adam)", "",
           "| Model | Total params | Trainable params | MACs / image | Size (KB, float32) | Size (KB, .keras file) | Epoch time (s) | Test acc | Macro precision | Macro recall |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ["A_adam", "B_adam"]:
        if tag in rows:
            m = rows[tag]
            out.append(f"| {tag[0]} | {m['total_params']:,} | {m['trainable_params']:,} | {m['macs']:,} | "
                       f"{m['size_kb_float32_estimate']:.1f} | {m['size_kb_keras_file']:.1f} | "
                       f"{m['mean_epoch_time_s']:.1f} | {m['test_accuracy']:.4f} | "
                       f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} |")

    out += ["", "## Optimizer comparison", "",
            "| Run | Best epoch | Best val loss | Best val acc | Test acc |",
            "|---|---|---|---|---|"]
    for tag, m in rows.items():
        out.append(f"| {tag} | {m['best_epoch']} | {m['best_val_loss']:.4f} | "
                   f"{m['best_val_accuracy']:.4f} | {m['test_accuracy']:.4f} |")

    if sota:
        out += ["", "## Model B vs fine-tuned lightweight SOTA", "",
                "| Model | Total params | Size (MB) | Epoch time (s) | Test acc | Macro precision | Macro recall |",
                "|---|---|---|---|---|---|---|"]
        if "B_adam" in rows:
            m = rows["B_adam"]
            out.append(f"| Model B | {m['total_params']:,} | {m['size_kb_float32_estimate'] / 1024:.3f} | "
                       f"{m['mean_epoch_time_s']:.1f} | {m['test_accuracy']:.4f} | "
                       f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} |")
        for name, m in sota.items():
            out.append(f"| {name} | {m['total_params']:,} | {m['size_mb_float32_estimate']:.2f} | "
                       f"{m['mean_epoch_time_s_phase2']:.1f} | {m['test_accuracy']:.4f} | "
                       f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} |")

    path = os.path.join("results", "comparison.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nSaved {path}")


def main():
    # Usage: python src/evaluate.py [tag ...]   e.g. A_adam B_adam   (default: every model in models/)
    tags = sys.argv[1:] or sorted(os.path.basename(p)[:-len(".keras")]
                                  for p in glob.glob(os.path.join("models", "*.keras")))
    _, _, test = get_datasets()
    for tag in tags:
        evaluate(tag, test)
    write_tables()


if __name__ == "__main__":
    main()
