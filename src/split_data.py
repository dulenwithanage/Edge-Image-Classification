import os, random, csv

random.seed(42)
roots = [
    os.path.join("data", "seg_train", "seg_train"),
    os.path.join("data", "seg_test", "seg_test"),
]
classes = sorted(os.listdir(roots[0]))

rows = []
for label, cls in enumerate(classes):
    files = []
    for r in roots:
        d = os.path.join(r, cls)
        files += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    random.shuffle(files)
    n = len(files)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    for i, p in enumerate(files):
        if i < n_train:
            subset = "train"
        elif i < n_train + n_val:
            subset = "val"
        else:
            subset = "test"
        rows.append((p, label, cls, subset))

with open(os.path.join("results", "split.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "label", "class", "subset"])
    w.writerows(rows)

for cls in classes:
    c = {s: sum(1 for r in rows if r[2] == cls and r[3] == s) for s in ["train", "val", "test"]}
    print(f"{cls:10s} train {c['train']}  val {c['val']}  test {c['test']}")
print("total", {s: sum(1 for r in rows if r[3] == s) for s in ["train", "val", "test"]})