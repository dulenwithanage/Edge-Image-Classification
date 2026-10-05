import os

root = "data"
for split in ["seg_train", "seg_test"]:
    split_dir = os.path.join(root, split, split)
    print(split)
    total = 0
    for cls in sorted(os.listdir(split_dir)):
        n = len(os.listdir(os.path.join(split_dir, cls)))
        total += n
        print(f"  {cls:10s} {n}")
    print(f"  total      {total}\n")