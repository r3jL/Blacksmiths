"""Shared config: classes, encoders, zero-shot prompts, embedding helper."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EMB = ROOT / "emb"

CLASSES = ["road_damage", "garbage", "water_issue", "streetlight", "pollution", "not_civic"]

# Frozen pretrained encoders (open_clip names, verified with open_clip.list_pretrained()).
ENCODERS = {
    "clip_b32": ("ViT-B-32", "openai"),
    "siglip_b16": ("ViT-B-16-SigLIP", "webli"),
}

PROMPTS = {
    "road_damage": ["a photo of a pothole on a road", "a photo of a damaged, cracked road",
                    "a photo of a broken road surface"],
    "garbage": ["a photo of a garbage dump on the street", "a photo of an overflowing garbage bin",
                "a photo of litter and trash on the roadside"],
    "water_issue": ["a photo of a flooded, waterlogged street", "a photo of a burst water pipe leaking water",
                    "a photo of a road under water after rain"],
    "streetlight": ["a photo of a street light", "a photo of a street lamp on a pole",
                    "a photo of a street light at night"],
    "pollution": ["a photo of smoke from burning garbage", "a photo of air pollution and smog",
                  "a photo of a polluted river with foam"],
    "not_civic": ["a photo of a person", "a selfie", "a photo of food", "a screenshot of a phone screen",
                  "a photo of an indoor room", "a photo of a clean road", "a photo of a building",
                  "a photo of a car", "a meme", "a photo of a shop"],
}


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(key):
    import open_clip
    arch, pretrained = ENCODERS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.eval().to(device())
    return model, preprocess, open_clip.get_tokenizer(arch)


@torch.no_grad()
def embed_images(model, preprocess, images, batch=32, flip=False):
    """L2-normalised image embeddings for a list of PIL images."""
    out = []
    for i in range(0, len(images), batch):
        ims = images[i:i + batch]
        if flip:
            ims = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims]
        x = torch.stack([preprocess(im) for im in ims]).to(device())
        f = model.encode_image(x).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def text_class_embeddings(model, tokenizer):
    """One normalised text embedding per class (mean over its prompts)."""
    rows = []
    for c in CLASSES:
        t = tokenizer(PROMPTS[c]).to(device())
        f = torch.nn.functional.normalize(model.encode_text(t).float(), dim=-1).mean(0)
        rows.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
    return np.stack(rows)


def zero_shot_probs(img_emb, txt_emb, scale=100.0):
    logits = scale * img_emb @ txt_emb.T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)
