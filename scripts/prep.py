import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
remap = {1: 0, 2: 1, 3: 2}  # elephant, rhino, zebra
for split in ["train", "val", "test"]:
    os.makedirs(f"{dst}/images/{split}", exist_ok=True); os.makedirs(f"{dst}/labels/{split}", exist_ok=True)
    kept = drop = 0; cnt = [0, 0, 0]
    for lf in os.listdir(f"{src}/labels/{split}"):
        lines = [l.split() for l in open(f"{src}/labels/{split}/{lf}") if l.strip()]
        cls = {int(l[0]) for l in lines}
        if not lines or 0 in cls: drop += 1; continue
        stem = lf[:-4]; img = next(f for f in os.listdir(f"{src}/images/{split}") if f.startswith(stem + ".")) if False else None
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            if os.path.exists(f"{src}/images/{split}/{stem}{ext}"): img = stem + ext; break
        if img is None: drop += 1; continue
        shutil.copy(f"{src}/images/{split}/{img}", f"{dst}/images/{split}/{img}")
        with open(f"{dst}/labels/{split}/{lf}", "w") as f:
            for l in lines:
                c = remap[int(l[0])]; cnt[c] += 1; f.write(" ".join([str(c)] + l[1:]) + "\n")
        kept += 1
    print(split, "images kept", kept, "dropped", drop, "boxes(ele,rhino,zebra)", cnt)
open(f"{dst}/data.yaml", "w").write(f"path: {dst}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: elephant\n  1: rhino\n  2: zebra\n")
