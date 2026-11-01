"""Steps 4-5: train the classifier head, calibrate, choose thresholds, save artifacts.

All choices use out-of-fold (OOF) predictions from group-aware 5-fold CV on the
train split, so test/demo stay untouched.
  1. Tune logistic-regression C.
  2. Optionally blend with zero-shot text-prompt probabilities (weight w).
  3. Temperature-scale the blended probabilities (minimise NLL).
  4. Pick the not_civic rejection bar and the human-review confidence bar.
Usage: python train.py --encoder clip_b32
"""
import argparse
import json
import platform

import joblib
import numpy as np
import open_clip
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import CLASSES, ENCODERS, ROOT
from compare import load, oof_probs

NC = CLASSES.index("not_civic")
# Conservative on purpose: these numbers come from web photos, real night-time
# phone photos will be harder.
MAX_CIVIC_REJECT = 0.005  # at most 0.5% of real complaints may be wrongly rejected by the gate
TARGET_AUTO_ACC = 0.95    # auto-accepted predictions should be at least this accurate


def temper(p, T):
    logp = np.log(np.clip(p, 1e-9, 1)) / T
    logp -= logp.max(1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(1, keepdims=True)


def macro_f1(y, p):
    return f1_score(y, p.argmax(1), labels=range(len(CLASSES)), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="clip_b32", choices=list(ENCODERS))
    args = ap.parse_args()
    tr, x, xf, zs = load(args.encoder)
    y = tr["label"].map(CLASSES.index).values
    groups = tr["group"].values

    print("C       OOF macro-F1")
    scores = {}
    for C in [0.1, 0.3, 1, 3, 10]:
        scores[C] = macro_f1(y, oof_probs(x, xf, y, groups, C=C))
        print(f"{C:<7} {scores[C]:.3f}")
    best = max(scores.values())
    C = min(c for c, s in scores.items() if s >= best - 0.005)  # simplest within 0.005 of best
    p_lr = oof_probs(x, xf, y, groups, C=C)

    print("\nzero-shot blend weight  OOF macro-F1")
    blend = {}
    for w in [0.0, 0.25, 0.5]:
        blend[w] = macro_f1(y, (1 - w) * p_lr + w * zs)
        print(f"{w:<23} {blend[w]:.3f}")
    print(f"zero-shot alone         {macro_f1(y, zs):.3f}")
    w = min(k for k, s in blend.items() if s >= max(blend.values()) - 0.005)
    p = (1 - w) * p_lr + w * zs

    Ts = np.linspace(0.2, 3, 57)
    nll = [-np.log(temper(p, T)[np.arange(len(y)), y] + 1e-9).mean() for T in Ts]
    T = float(Ts[int(np.argmin(nll))])
    pc = temper(p, T)
    print(f"\nchosen C={C}, zero-shot weight={w}, temperature={T:.2f} (NLL {min(nll):.3f})")

    # Gate: reject as not_civic only when p(not_civic) >= t_reject.
    civic = y != NC
    lines = ["| t_reject | real complaints wrongly rejected | not_civic correctly rejected |", "|---|---|---|"]
    t_reject = 0.99
    for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        rej = pc[:, NC] >= t
        fr, tr_ = rej[civic].mean(), rej[~civic].mean()
        lines.append(f"| {t} | {fr:.1%} ({rej[civic].sum()}/{civic.sum()}) | {tr_:.1%} ({rej[~civic].sum()}/{(~civic).sum()}) |")
        if fr <= MAX_CIVIC_REJECT and t < t_reject:
            t_reject = t
    gate_table = "\n".join(lines)

    # Review: send to humans when top probability < t_review.
    conf, pred = pc.max(1), pc.argmax(1)
    lines = ["| t_review | auto-handled share | accuracy of auto-handled | sent to review |", "|---|---|---|---|"]
    t_review = 0.9
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        auto = conf >= t
        acc = (pred[auto] == y[auto]).mean() if auto.any() else float("nan")
        lines.append(f"| {t} | {auto.mean():.1%} | {acc:.1%} | {(~auto).sum()} |")
        if acc >= TARGET_AUTO_ACC and t < t_review:
            t_review = t
    review_table = "\n".join(lines)
    print("\nGate (not_civic rejection):\n" + gate_table)
    print("\nHuman review threshold:\n" + review_table)
    print(f"\nchosen t_reject={t_reject}, t_review={t_review}")

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit(np.vstack([x, xf]), np.concatenate([y, y]))
    z = np.load(ROOT / "emb" / f"{args.encoder}.npz", allow_pickle=True)
    arch, pretrained = ENCODERS[args.encoder]
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    joblib.dump({"coef": clf.coef_, "intercept": clf.intercept_, "text_emb": z["txt"],
                 "zs_weight": w, "temperature": T}, art / "head.joblib")
    meta = {
        "classes": CLASSES, "encoder_key": args.encoder, "open_clip_arch": arch,
        "open_clip_pretrained": pretrained, "C": C, "zero_shot_weight": w, "temperature": T,
        "t_reject_not_civic": t_reject, "t_review": t_review, "n_train": int(len(y)),
        "train_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "oof_macro_f1": round(float(macro_f1(y, pc)), 4),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "open_clip": open_clip.__version__, "sklearn": sklearn.__version__},
    }
    (art / "meta.json").write_text(json.dumps(meta, indent=2))
    (art / "thresholds.md").write_text(
        f"# Threshold trade-offs (out-of-fold, train split, n={len(y)})\n\n"
        f"Chosen: t_reject={t_reject} (wrongly reject <= {MAX_CIVIC_REJECT:.0%} of real complaints), "
        f"t_review={t_review} (auto-handled accuracy >= {TARGET_AUTO_ACC:.0%}).\n\n"
        f"## Gate\n\n{gate_table}\n\n## Review\n\n{review_table}\n")
    print("saved artifacts/head.joblib, meta.json, thresholds.md")


if __name__ == "__main__":
    main()
