"""Compare classifier vs hosted VLM on the same test photos.

Input: a CSV with columns  path,vlm_label  (path as in data/split.csv, label one
of the 6 classes). Run evaluate.py first; it writes data/eval_test_predictions.csv.
Usage: python compare_vlm.py vlm_predictions.csv
"""
import sys

import pandas as pd

from common import DATA


def main():
    clf = pd.read_csv(DATA / "eval_test_predictions.csv")
    vlm = pd.read_csv(sys.argv[1])
    df = clf.merge(vlm, on="path", how="inner")
    agree = df["pred"] == df["vlm_label"]
    clf_ok, vlm_ok = df["pred"] == df["label"], df["vlm_label"] == df["label"]
    print(f"n={len(df)}  agreement={agree.mean():.1%}")
    print(f"classifier accuracy {clf_ok.mean():.1%}  VLM accuracy {vlm_ok.mean():.1%}")
    print(f"when they agree: correct {clf_ok[agree].mean():.1%} of the time")
    d = df[~agree]
    if len(d):
        print(f"when they disagree ({len(d)}): classifier right {clf_ok[~agree].sum()}, "
              f"VLM right {vlm_ok[~agree].sum()}, neither {(~clf_ok[~agree] & ~vlm_ok[~agree]).sum()}")
        print(d[["path", "label", "pred", "vlm_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
