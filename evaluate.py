"""Step 6: run the final model ONCE on the untouched test split and the demo set.

Prints accuracy, macro-F1 (with 95% bootstrap intervals), per-class precision /
recall, confusion matrix, gate and review behaviour, and saves a contact sheet
of mistakes to data/eval_<split>_mistakes.jpg.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import classifier
from common import CLASSES, DATA
from embed import contact_sheet


def bootstrap_ci(y, pred, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def main():
    split = pd.read_csv(DATA / "split.csv")
    labels = list(range(len(CLASSES)))
    for name in ["test", "demo"]:
        df = split[split["split"] == name].reset_index(drop=True)
        if df.empty:
            print(f"== {name}: no images yet")
            continue
        out = [classifier.predict((DATA / p).read_bytes()) for p in df["path"]]
        y = df["label"].map(CLASSES.index).values
        pred = np.array([CLASSES.index(o["label"]) for o in out])
        acc = (y == pred).mean()
        f1 = lambda a, b: f1_score(a, b, labels=labels, average="macro", zero_division=0)
        acc_ci = bootstrap_ci(y, pred, lambda a, b: (a == b).mean())
        f1_ci = bootstrap_ci(y, pred, f1)
        print(f"\n== {name}: n={len(y)}")
        print(f"accuracy {acc:.1%} (95% CI {acc_ci[0]:.0%}-{acc_ci[1]:.0%}), "
              f"macro-F1 {f1(y, pred):.3f} (95% CI {f1_ci[0]:.2f}-{f1_ci[1]:.2f})")
        print(classification_report(y, pred, labels=labels, target_names=CLASSES, zero_division=0))
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"true {c}" for c in CLASSES],
                          columns=[c[:8] for c in CLASSES])
        print(cm.to_string())

        civic = y != CLASSES.index("not_civic")
        rejected = np.array([not o["is_civic"] for o in out])
        review = np.array([o["needs_review"] for o in out])
        print(f"\ngate: wrongly rejected {rejected[civic].sum()}/{civic.sum()} real complaints, "
              f"rejected {rejected[~civic].sum()}/{(~civic).sum()} not_civic")
        auto = ~review
        if auto.any():
            print(f"review: {review.sum()}/{len(y)} sent to humans; "
                  f"accuracy of auto-handled {(pred[auto] == y[auto]).mean():.1%}")

        df["pred"] = [o["label"] for o in out]
        df["confidence"] = [o["confidence"] for o in out]
        df.to_csv(DATA / f"eval_{name}_predictions.csv", index=False)
        wrong = df[df["label"] != df["pred"]].sort_values("confidence", ascending=False)
        if len(wrong):
            contact_sheet(wrong["path"].tolist(), [f"{a}>{b} {c:.2f}" for a, b, c in
                                                    zip(wrong.label, wrong.pred, wrong.confidence)],
                          DATA / f"eval_{name}_mistakes.jpg")
            print(f"{len(wrong)} mistakes -> data/eval_{name}_mistakes.jpg")


if __name__ == "__main__":
    main()
