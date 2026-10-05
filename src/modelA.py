import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
import tensorflow as tf
from tensorflow.keras import layers, models
 
IMG_SIZE = (64, 64)
NUM_CLASSES = 6
 
 
def build_model_a():
    """Model A: standard 2D convolutions interleaved with max-pooling."""
    m = models.Sequential(name="model_a_standard")
    m.add(layers.Input(shape=(*IMG_SIZE, 3)))
 
    # Four conv blocks: Conv -> BatchNorm -> ReLU -> MaxPool
    for filters in [32, 64, 128, 128]:
        m.add(layers.Conv2D(filters, 3, padding="same", use_bias=True))
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())
        m.add(layers.MaxPooling2D(2))
 
    # Global average pooling avoids a large flatten + dense layer
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(64))
    m.add(layers.ReLU())
    m.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return m
 
 
if __name__ == "__main__":
    model = build_model_a()
    model.summary()