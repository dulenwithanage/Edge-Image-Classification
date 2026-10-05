import os
from collections import Counter
from PIL import Image

split_dir = os.path.join("data", "seg_train", "seg_train")
sizes = Counter()

for cls in os.listdir(split_dir):
    for f in os.listdir(os.path.join(split_dir, cls)):
        with Image.open(os.path.join(split_dir, cls, f)) as img:
            sizes[(img.size, img.mode)] += 1

for k, v in sizes.most_common():
    print(k, v)

print("script started")