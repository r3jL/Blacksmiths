"""Inference: frozen pretrained image encoder + our trained classifier head.

predict(image_bytes) -> {label, probabilities, confidence, is_civic, needs_review, embedding, ...}
"""
import io
import json
from functools import lru_cache

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from common import ROOT, device

ART = ROOT / "artifacts"


@lru_cache(maxsize=1)
def _load():
    import open_clip
    meta = json.loads((ART / "meta.json").read_text())
    head = joblib.load(ART / "head.joblib")
    model, _, preprocess = open_clip.create_model_and_transforms(
        meta["open_clip_arch"], pretrained=meta["open_clip_pretrained"])
    return meta, head, model.eval().to(device()), preprocess


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@torch.no_grad()
def embed(image_bytes):
    meta, head, model, preprocess = _load()
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    x = preprocess(im).unsqueeze(0).to(device())
    f = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
    return f[0].cpu().numpy()


def predict(image_bytes):
    meta, head, _, _ = _load()
    classes = meta["classes"]
    e = embed(image_bytes)
    p_head = _softmax(head["coef"] @ e + head["intercept"])
    p_zs = _softmax(100.0 * head["text_emb"] @ e)
    w = head["zs_weight"]
    p = (1 - w) * p_head + w * p_zs
    p = _softmax(np.log(np.clip(p, 1e-9, 1)) / head["temperature"])

    k = int(p.argmax())
    p_nc = float(p[classes.index("not_civic")])
    is_civic = p_nc < meta["t_reject_not_civic"]
    # Borderline not_civic images are passed on (not rejected) and flagged for review.
    needs_review = bool(p[k] < meta["t_review"] or (classes[k] == "not_civic" and is_civic))
    civic_probs = {c: float(v) for c, v in zip(classes, p) if c != "not_civic"}
    return {
        "label": classes[k],
        "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)},
        "confidence": round(float(p[k]), 4),
        "best_civic_label": max(civic_probs, key=civic_probs.get),
        "is_civic": bool(is_civic),
        "needs_review": needs_review,
        "embedding": [round(float(v), 6) for v in e],
        "model": f"{meta['open_clip_arch']}/{meta['open_clip_pretrained']} frozen + logistic-regression head",
    }


def cosine(a, b):
    """Similarity of two returned embeddings; > ~0.92 suggests the same scene (tune on real reports)."""
    return float(np.dot(a, b))
