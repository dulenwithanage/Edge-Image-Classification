import os
import matplotlib.pyplot as plt


def plot_history(history, name):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, h["loss"], "b-o", markersize=3, label="Training loss")
    ax[0].plot(epochs, h["val_loss"], "r-o", markersize=3, label="Validation loss")
    ax[0].set(title=f"{name}: loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax[0].legend()

    ax[1].plot(epochs, h["accuracy"], "b-o", markersize=3, label="Training acc")
    ax[1].plot(epochs, h["val_accuracy"], "r-o", markersize=3, label="Validation acc")
    ax[1].set(title=f"{name}: accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{name}_curves.png"), dpi=150)
    plt.show()