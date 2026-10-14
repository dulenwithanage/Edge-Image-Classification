import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
import tensorflow as tf
from tensorflow.keras import layers, models

from data import augmentation
 
IMG_SIZE = (64, 64)
NUM_CLASSES = 6
 
 
def build_model_a():
    """Model A: standard 2D convolutions interleaved with max-pooling."""
    m = models.Sequential(name="model_a_standard")
    m.add(layers.Input(shape=(*IMG_SIZE, 3)))
    m.add(augmentation())  # active during training only
 
    # Four conv blocks: Conv -> BatchNorm -> ReLU -> MaxPool
    for filters in [32, 64, 128, 128]:
        m.add(layers.Conv2D(filters, 3, padding="same", use_bias=True))
        # momentum=0.9 (default 0.99) lets the running mean/variance used at
        # validation time keep up with the weights, which reduces val-loss spikes
        m.add(layers.BatchNormalization(momentum=0.9))
        m.add(layers.ReLU())
        m.add(layers.MaxPooling2D(2))
 
    # Global average pooling avoids a large flatten + dense layer
    m.add(layers.GlobalAveragePooling2D())
    # Dropout randomly zeroes 30% of the pooled features each training step,
    # so the classifier cannot rely on any single feature (off at test time)
    m.add(layers.Dropout(0.3))
    m.add(layers.Dense(64))
    m.add(layers.ReLU())
    m.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return m
 
 
if __name__ == "__main__":
    model = build_model_a()
    model.summary()