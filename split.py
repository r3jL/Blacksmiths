"""Step 2c: group-aware split. Writes data/split.csv (path, label, source, group, split).

- web photos          -> train (never test)
- own photos          -> split by spot group: --own-train-frac of each class's
                         spots go to train, the rest to test. All photos of one
                         spot (and any near-duplicates) stay together.
- webtest photos      -> test (held-out Commons photos: other queries, other photographers)
- demo photos         -> demo (separate live-demo simulation set)
Training images whose CLIP embedding nearly matches a test/demo image
(cosine >= 0.95) are dropped, on top of the perceptual-hash grouping.
Model selection uses group-aware cross-validation inside train only; test and
demo are touched exactly once, by evaluate.py.
"""
import argparse
import math

import numpy as np
import pandas as pd

from common import CLASSES, DATA, EMB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own-train-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    df = pd.read_csv(DATA / "index.csv")
    df["split"] = np.where(df["source"] == "demo", "demo", "train")
    own = df[df["source"] == "own"]
    # A group belongs to the class most of its photos have.
    group_label = own.groupby("group")["label"].agg(lambda s: s.mode()[0])
    rng = np.random.default_rng(args.seed)
    test_groups = set()
    for cls in CLASSES:
        groups = sorted(group_label[group_label == cls].index)
        rng.shuffle(groups)
        n_train = math.floor(args.own_train_frac * len(groups))
        test_groups |= set(groups[n_train:])
    test_groups |= set(df.loc[df["source"] == "webtest", "group"])
    df.loc[df["group"].isin(test_groups), "split"] = "test"
    # A web image near-duplicating a test photo shares its group, so it leaves train too.
    leaked = df["group"].isin(test_groups) & (df["source"] == "web")
    emb_file = EMB / "clip_b32.npz"
    if emb_file.exists():
        z = np.load(emb_file, allow_pickle=True)
        pos = {p: i for i, p in enumerate(z["paths"])}
        idx = df["path"].map(pos)
        held = (df["split"] != "train").values & idx.notna().values
        train = (df["split"] == "train").values & idx.notna().values
        if held.any() and train.any():
            sim = z["x"][idx[train].astype(int)] @ z["x"][idx[held].astype(int)].T
            close = np.zeros(len(df), bool)
            close[np.flatnonzero(train)[sim.max(1) >= 0.95]] = True
            print(f"embedding near-duplicates of test/demo dropped from train: {close.sum()}")
            leaked |= close
    df = df[~leaked]

    df[["path", "label", "source", "group", "split"]].to_csv(DATA / "split.csv", index=False)
    print(pd.crosstab([df["split"], df["source"]], df["label"], margins=True))
    if leaked.any():
        print(f"removed {leaked.sum()} web images that near-duplicate test photos")
    overlap = set(df[df.split == "train"].group) & set(df[df.split == "test"].group)
    print("group overlap train/test:", len(overlap))


if __name__ == "__main__":
    main()
