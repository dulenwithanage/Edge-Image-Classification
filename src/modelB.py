import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (64, 64)
NUM_CLASSES = 6


def build_model_b():
    """Model B: depthwise separable convolutions (lightweight, < 100k params)."""
    m = models.Sequential(name="model_b_lightweight")
    m.add(layers.Input(shape=(*IMG_SIZE, 3)))

    # All four blocks use depthwise separable convolutions
    # (3x3 depthwise per channel, then 1x1 pointwise to mix channels)
    for filters in [32, 64, 128, 128]:
        m.add(layers.SeparableConv2D(filters, 3, padding="same", use_bias=False))
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())
        m.add(layers.MaxPooling2D(2))

    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(64))
    m.add(layers.ReLU())
    m.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return m


if __name__ == "__main__":
    model = build_model_b()
    model.summary()
    n = model.count_params()
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"Trainable: {trainable}  (limit 100,000) -> {'OK' if trainable <= 100000 else 'TOO BIG'}")