"""Step 2b/3a: embed every indexed image with each encoder and cache to emb/.

Also writes zero-shot probabilities and, for web images, a list of likely junk
(zero-shot strongly disagrees with the folder label) plus a contact sheet.

Usage: python embed.py [clip_b32 siglip_b16]
"""
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import CLASSES, DATA, EMB, ENCODERS, embed_images, load_encoder, text_class_embeddings, zero_shot_probs


def contact_sheet(paths, captions, out, thumb=160, cols=8):
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (p, cap) in enumerate(zip(paths, captions)):
        im = Image.open(DATA / p).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 28)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + thumb + 2), cap[:26], fill="black")
        d.text((x + 2, y + thumb + 14), f"#{k}", fill="red")
    sheet.save(out, quality=85)


def main():
    keys = sys.argv[1:] or list(ENCODERS)
    df = pd.read_csv(DATA / "index.csv")
    EMB.mkdir(exist_ok=True)
    images = [Image.open(DATA / p).convert("RGB") for p in df["path"]]
    for key in keys:
        t0 = time.time()
        model, preprocess, tok = load_encoder(key)
        x = embed_images(model, preprocess, images)
        x_flip = embed_images(model, preprocess, images, flip=True)
        txt = text_class_embeddings(model, tok)
        zs = zero_shot_probs(x, txt)
        np.savez(EMB / f"{key}.npz", x=x, x_flip=x_flip, txt=txt, zs=zs, paths=df["path"].values)
        y = df["label"].map(CLASSES.index).values
        acc = (zs.argmax(1) == y).mean()
        print(f"{key}: dim={x.shape[1]} n={len(x)} time={time.time() - t0:.0f}s "
              f"zero-shot acc on all indexed images={acc:.3f}")

        if key == keys[0]:
            # Likely junk: web image whose own-label zero-shot prob is tiny.
            own_p = zs[np.arange(len(y)), y]
            flag = (df["source"] == "web").values & (own_p < 0.05) & (zs.argmax(1) != y)
            j = df[flag].copy()
            j["zs_pred"] = [CLASSES[k] for k in zs[flag].argmax(1)]
            j["own_label_prob"] = own_p[flag].round(3)
            j = j.sort_values(["label", "own_label_prob"]).reset_index(drop=True)
            j.to_csv(DATA / "junk_candidates.csv", index_label="sheet_id")
            if len(j):
                contact_sheet(j["path"].tolist(), [f"{a}>{b}" for a, b in zip(j.label, j.zs_pred)],
                              DATA / "junk_candidates.jpg")
            print(f"  flagged {len(j)} web images as possible junk -> data/junk_candidates.csv/.jpg")


if __name__ == "__main__":
    main()
