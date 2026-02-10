from pathlib import Path
from scipy.io import loadmat
import shutil

repo = Path.cwd()
src_dir = repo / "assets" / "oxford102" / "jpg"
out_dir = repo / "assets" / "flower_data"

labels = loadmat(repo / "assets" / "oxford102" / "imagelabels.mat")["labels"].squeeze()
setid  = loadmat(repo / "assets" / "oxford102" / "setid.mat")

splits = {
    "train": setid["trnid"].squeeze(),
    "valid": setid["valid"].squeeze(),
    "test":  setid["tstid"].squeeze(),
}

# rebuild output dir cleanly
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# create class folders
for split in splits:
    for c in range(1, 103):
        (out_dir / split / str(c)).mkdir(parents=True, exist_ok=True)

def img_path(img_id: int) -> Path:
    return src_dir / f"image_{img_id:05d}.jpg"

count = 0
missing = 0

for split, ids in splits.items():
    for img_id in ids:
        img_id = int(img_id)
        cls = int(labels[img_id - 1])  # labels indexed 1..8189
        src = img_path(img_id)
        if not src.exists():
            missing += 1
            continue
        dst = out_dir / split / str(cls) / src.name
        shutil.copy2(src, dst)
        count += 1

print("Done. Total copied:", count, "missing:", missing)
for split in ("train","valid","test"):
    n = sum(1 for _ in (out_dir / split).rglob("*.jpg"))
    print(f"{split}: {n}")
