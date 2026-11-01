# Civic image classifier (Forge'26)

**What it is, in one line:** a frozen pretrained image encoder (OpenAI CLIP ViT-B/32, via `open_clip`) plus a logistic-regression classifier head that we trained during the hackathon. It is *not* a custom neural network and *not* a trained VLM.

## What it does in the system
The hosted VLM is the main engine. This classifier:
1. **Gate:** rejects clearly irrelevant images (selfies, food, screenshots, indoor) before we spend a VLM call.
2. **Cross-check:** predicts a category. If it disagrees with the VLM or is unsure, the report goes to human review.
3. **Fallback:** gives a category if the VLM times out.
4. **Duplicates:** returns a 512-d image embedding; cosine similarity between two reports at the same location flags near-duplicates.

Severity is **not** predicted here (it comes from a fixed formula on the VLM's output).

## Classes
`road_damage`, `garbage`, `water_issue` (waterlogging + leaks merged), `streetlight` (any photo of a street light; whether it is broken is decided by the VLM and the citizen's text, since a broken lamp looks the same as a working one by day), `pollution` (smoke, burning waste, polluted water), `not_civic`. Noise is out of scope (no image signature).

## Data
| source | use | count |
|---|---|---|
| Wikimedia Commons, downloaded with `scrape_wikimedia.py` (Commons API only; CC0 / public domain / CC BY / CC BY-SA, JPEG photos, AI-generated categories excluded) | training only | 833 after cleaning (130 to 146 per class) |
| Wikimedia Commons held-out set (`--set test`: different queries, photographers in the training set excluded), labels checked by eye | test only | 104 |
| Our own phone photos, `data/own/<class>/<class>__<spot>__<initials>__<n>.jpg` | test set (and a share of spots for training) | none yet |
| Fresh demo-style photos, `data/demo/<class>/` | separate demo check | ~20 |

* Every Commons file's title, author, license and URL is in **`data/web_manifest.csv`**, our disclosure file.
* 43 off-topic Commons images (drawings, diagrams, lab photos, ambiguous street scenes) were removed after manual review; they are listed in `data/rejected/rejected_log.csv`.
* No synthetic or generated images are used.

**Cleaning (`check_data.py`):** converts to RGB JPEG (max side 768), drops corrupt, tiny (< 150 px) and exact-duplicate files. Near-duplicates (perceptual hash) are merged into one group.

**Leakage control (`split.py`):** photos of the same spot (from the filename) and near-duplicates always stay in the same split. The test set contains only our own photos and is used once, by `evaluate.py`.

## Method
1. Embed each image with frozen CLIP ViT-B/32 (`open_clip`: `ViT-B-32`, pretrained `openai`). Horizontal flips are added as cheap augmentation for training.
2. Encoder choice (`compare.py`, 5-fold group-aware stratified CV, logistic regression):

   | encoder | macro-F1 (trained head) | macro-F1 (zero-shot prompts) |
   |---|---|---|
   | CLIP ViT-B/32 | 0.892 | 0.789 |
   | SigLIP ViT-B/16 | 0.897 | 0.766 |

   The difference is within noise, so we picked CLIP: 2.5× faster and smaller. The trained head beats zero-shot by about 10 points.
3. Head (`train.py`): logistic regression, `class_weight="balanced"`, C = 3 tuned by CV. Blending with zero-shot did not help, so it is not used. Temperature scaling (T = 0.55) calibrates the probabilities. Cross-validated (out-of-fold) macro-F1 is **0.904** on web photos.
4. Thresholds (`artifacts/thresholds.md` has the full trade-off tables):
   * **Reject as not_civic only if p(not_civic) ≥ 0.8.** We would rather send a borderline image to the VLM than reject a real complaint. Out of fold, this wrongly rejects 0.4% of real complaints and catches 87% of irrelevant images.
   * **Human review if top confidence < 0.7.** Out of fold, this sends 13% to review, and the rest are 95% accurate.

## Evaluation
**Held-out web test set (run once, 2026-11-01 20:23 IST):** 104 Commons photos, kept separate from training by using different search queries (other Indian cities) and excluding every photographer who appears in the training set. Each label was checked by eye; 136 of 240 downloads were rejected (`data/rejected_test/`). Sources are in `data/webtest_manifest.csv`. These are **not** our own Delhi photos.

| | result |
|---|---|
| accuracy | **83.7%** (95% bootstrap CI 76–90%) |
| macro-F1 | 0.81 (CI 0.71–0.89) |
| zero-shot CLIP prompts, same photos | 88.5% (it scored lower in cross-validation, and the CIs overlap) |
| gate (not_civic at p ≥ 0.8) | wrongly rejected **0 of 67** real complaints; caught 15 of 37 irrelevant images |
| human review (confidence < 0.7) | 25 of 104 sent to review; auto-handled ones 88.6% correct |

Per-class recall: road_damage 13/13, garbage 9/9, streetlight 21/22, water_issue 12/15, not_civic 28/37, pollution 4/8 (weakest).

Typical mistakes (`data/eval_test_mistakes.jpg`):
* River foam labelled `pollution` is predicted `water_issue`: "dirty water" sits between the two classes.
* Buildings and night cityscapes are predicted `streetlight`.
* People and indoor offices are predicted `garbage`.
* A garbage-filled drain we labelled `water_issue` was predicted `garbage`, which is arguably also right.

**Pitch sentence:** "On 104 held-out, openly licensed real photos (different photographers and cities from the training data), the classifier reached 84% accuracy (95% CI 76–90%). It never rejected a real complaint at the gate, and the weakest class was pollution."

Run `python evaluate.py` again after adding our own photos (`data/own/`) or demo photos (`data/demo/`). It reports accuracy and macro-F1 with 95% bootstrap intervals, per-class precision and recall, a confusion matrix, gate and review behaviour, and a contact sheet of mistakes. `compare_vlm.py vlm.csv` measures agreement with the VLM.

**Pitch template:** "On N held-out real photos we took in Delhi, the classifier reached X% accuracy (95% CI A–B%); the weakest class was Y."

**Do not claim:**
* that cross-validated web-photo numbers are real-world accuracy;
* anything per class from fewer than about 20 test photos;
* that it detects whether a streetlight is broken;
* that it "understands" severity.

## Limitations
* All training and test photos so far are Commons photos, mostly daytime and many not from India. Real citizen phone photos (often at night) may score lower.
* The web test labels were checked by eye from contact sheets in a first pass by our AI coding assistant. A teammate should spot-check `data/webtest/`. A few labels are judgement calls.
* Small dataset (about 140 per class).
* `garbage` vs `pollution` overlap (burning garbage).
* `water_issue` mixes floods and pipe leaks.

## Run it
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
./run_all.sh                                  # rebuild: check -> embed -> split -> compare -> train -> evaluate
uvicorn api:app --host 0.0.0.0 --port 8000    # serve
curl -F "file=@photo.jpg" http://localhost:8000/classify
```
The first start downloads the CLIP weights (~600 MB) into `~/.cache/huggingface`; the head itself is 36 KB (`artifacts/`).

## API
`POST /classify` (multipart field `file`, up to 10 MB) returns:
```json
{"label": "garbage", "probabilities": {"road_damage": 0.01, "...": 0.0}, "confidence": 0.97,
 "best_civic_label": "garbage", "is_civic": true, "needs_review": false,
 "embedding": [512 floats], "model": "ViT-B-32/openai frozen + logistic-regression head"}
```
On any error or timeout (10 s), it returns a non-200 status with `{"error": ..., "fallback": "vlm_only"}`. The backend should then continue with the VLM alone. `GET /health` checks that the service is up.

Duplicate check: `cosine = dot(embedding_a, embedding_b)` (embeddings are L2-normalised). A value above about 0.92 for reports within ~50 m suggests the same scene. That threshold is a starting guess and should be tuned on real reports.

## Deployment notes
* The model needs about 1 GB RAM with PyTorch and runs at about 25 ms per image on an M1 (CPU-only hosts are slower).
* It is too big for Vercel's serverless Python functions (PyTorch alone exceeds their bundle limit, as far as we know). Run it on a laptop and expose it with `cloudflared tunnel --url http://localhost:8000`, or on a free VM with at least 1 GB RAM.
* Cold start is about 1 minute: the weights load at startup, not on the first request.
