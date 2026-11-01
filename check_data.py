"""Step 2a: validate, normalise and index all photos.

Reads  data/web/<class>/*  (Commons, training only)
       data/webtest/<class>/* (Commons, held out: other queries and photographers, test only)
       data/own/<class>/*  (our photos: <class>__<spot>__<initials>__<n>.jpg)
       data/demo/<class>/* (fresh demo-style photos, optional)
Writes data/clean/<source>/<class>/<name>.jpg (RGB JPEG, longest side <= 768)
       data/index.csv with one row per kept image and its leakage group.

Drops corrupt files, tiny images (< 150 px short side) and exact duplicates.
Near-duplicates (perceptual hash distance <= 6) are merged into one group so
they always land in the same split.
"""
import hashlib

import imagehash
import pandas as pd
from PIL import Image, ImageOps

from common import CLASSES, DATA

MAX_SIDE, MIN_SIDE, NEAR_DUP = 768, 150, 6
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}


def spot_of(source, cls, stem):
    if source in ("web", "webtest"):
        return f"{source}:{stem}"
    parts = stem.split("__")
    # <class>__<spot>__<initials>__<n>; fall back to the whole name if not followed
    return f"{source}:{parts[1]}" if len(parts) >= 3 else f"{source}:{stem}"


def main():
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    rows, problems = [], []
    seen_md5 = {}
    # Test sources first, so when a training web image duplicates one, the training copy is dropped.
    for source in ["own", "demo", "webtest", "web"]:
        for cls in CLASSES:
            folder = DATA / source / cls
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in EXTS:
                    continue
                try:
                    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                except Exception as e:
                    problems.append((str(p), f"corrupt: {e}"))
                    continue
                if min(im.size) < MIN_SIDE:
                    problems.append((str(p), f"tiny {im.size}"))
                    continue
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                md5 = hashlib.md5(im.tobytes()).hexdigest()
                if md5 in seen_md5:
                    problems.append((str(p), f"exact duplicate of {seen_md5[md5]}"))
                    continue
                seen_md5[md5] = str(p)
                dest = DATA / "clean" / source / cls / (p.stem + ".jpg")
                dest.parent.mkdir(parents=True, exist_ok=True)
                im.save(dest, "JPEG", quality=90)
                rows.append({"path": str(dest.relative_to(DATA)), "source": source, "label": cls,
                             "group": spot_of(source, cls, p.stem), "w": im.width, "h": im.height,
                             "phash": str(imagehash.phash(im))})

    df = pd.DataFrame(rows)
    # Union-find over near-duplicate pairs so they share a group.
    parent = {g: g for g in df["group"]}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    near = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if hashes[i] - hashes[j] <= NEAR_DUP:
                near.append((df.path[i], df.path[j]))
                parent[find(df.group[i])] = find(df.group[j])
    df["group"] = [find(g) for g in df["group"]]
    df.to_csv(DATA / "index.csv", index=False)
    pd.DataFrame(near, columns=["a", "b"]).to_csv(DATA / "near_duplicates.csv", index=False)
    pd.DataFrame(problems, columns=["file", "problem"]).to_csv(DATA / "problems.csv", index=False)

    print(pd.crosstab(df["label"], df["source"], margins=True))
    print(f"\ndropped {len(problems)} files (see data/problems.csv)")
    print(f"near-duplicate pairs merged into shared groups: {len(near)} (data/near_duplicates.csv)")
    cross = [(a, b) for a, b in near if a.split("/")[2] != b.split("/")[2]]
    if cross:
        print(f"WARNING: {len(cross)} near-duplicate pairs have different labels:")
        for a, b in cross[:10]:
            print("  ", a, "<->", b)


if __name__ == "__main__":
    main()
