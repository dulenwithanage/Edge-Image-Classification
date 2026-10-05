import os, random
import matplotlib.pyplot as plt
from PIL import Image

split_dir = os.path.join("data", "seg_train", "seg_train")
classes = sorted(os.listdir(split_dir))

fig, axes = plt.subplots(len(classes), 4, figsize=(8, 12))
for i, cls in enumerate(classes):
    files = random.sample(os.listdir(os.path.join(split_dir, cls)), 4)
    for j, f in enumerate(files):
        img = Image.open(os.path.join(split_dir, cls, f))
        axes[i, j].imshow(img)
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(cls, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(os.path.join("plots", "samples.png"), dpi=120)
plt.show()