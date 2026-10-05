import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import tensorflow as tf

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 6
AUTOTUNE = tf.data.AUTOTUNE


def read_split(csv_path=os.path.join("results", "split.csv")):
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            data[row["subset"]][0].append(row["path"])
            data[row["subset"]][1].append(int(row["label"]))
    return data


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(paths), seed=42)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def get_datasets():
    d = read_split()
    train = make_ds(*d["train"], training=True)
    val = make_ds(*d["val"], training=False)
    test = make_ds(*d["test"], training=False)
    return train, val, test


if __name__ == "__main__":
    train, val, test = get_datasets()
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        x, y = next(iter(ds))
        print(name, x.shape, x.dtype, float(tf.reduce_min(x)), float(tf.reduce_max(x)), y[:8].numpy())