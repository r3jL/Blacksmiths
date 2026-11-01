"""Step 3b: compare encoders with group-aware, stratified 5-fold CV on train only.

For each encoder: logistic regression (balanced) on frozen embeddings vs the
zero-shot text-prompt baseline. Reports macro-F1 and per-class recall, plus the
same numbers restricted to our own photos (the domain that matters).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASSES, DATA, EMB, ENCODERS

warnings.filterwarnings("ignore")


def load(key):
    z = np.load(EMB / f"{key}.npz", allow_pickle=True)
    split = pd.read_csv(DATA / "split.csv")
    pos = {p: i for i, p in enumerate(z["paths"])}
    tr = split[split["split"] == "train"].reset_index(drop=True)
    idx = tr["path"].map(pos).values
    return tr, z["x"][idx], z["x_flip"][idx], z["zs"][idx]


def oof_probs(x, x_flip, y, groups, C=1.0, seed=0):
    """Out-of-fold probabilities; training folds are augmented with flipped copies."""
    p = np.zeros((len(y), len(CLASSES)))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, va_i in cv.split(x, y, groups):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(np.vstack([x[tr_i], x_flip[tr_i]]), np.concatenate([y[tr_i], y[tr_i]]))
        p[va_i] = clf.predict_proba(x[va_i])
    return p


def report(name, y, pred, mask=None):
    if mask is not None:
        y, pred = y[mask], pred[mask]
    if len(y) == 0:
        return {}
    rec = recall_score(y, pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    row = {"model": name, "n": len(y), "acc": (y == pred).mean(),
           "macro_f1": f1_score(y, pred, labels=range(len(CLASSES)), average="macro", zero_division=0)}
    row.update({f"R_{c}": r for c, r in zip(CLASSES, rec)})
    return row


def main():
    keys = sys.argv[1:] or list(ENCODERS)
    rows = []
    for key in keys:
        tr, x, xf, zs = load(key)
        y = tr["label"].map(CLASSES.index).values
        own = (tr["source"] == "own").values
        p = oof_probs(x, xf, y, tr["group"].values)
        for name, pred in [(f"{key}+logreg", p.argmax(1)), (f"{key} zero-shot", zs.argmax(1))]:
            rows.append(report(name, y, pred))
            if own.any():
                rows.append(report(name + " [own only]", y, pred, own))
    out = pd.DataFrame([r for r in rows if r]).set_index("model")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
    out.to_csv(DATA / "encoder_comparison.csv")


if __name__ == "__main__":
    main()
