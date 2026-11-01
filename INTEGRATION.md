# Connecting the website to the classifier

**Base URL (changes if the tunnel restarts, so ask Rejul for the current one):**
`https://crop-neither-organize-independence.trycloudflare.com`

Put it in an environment variable (`CLASSIFIER_URL`), never hard-code it.

## The one endpoint you need
`POST /classify`, a `multipart/form-data` request with one field, `file` (the photo, max 10 MB, JPEG/PNG/HEIC).

```bash
curl -F "file=@pothole.jpg" $CLASSIFIER_URL/classify
```

Success (HTTP 200):
```json
{
  "label": "road_damage",            // one of: road_damage, garbage, water_issue, streetlight, pollution, not_civic
  "confidence": 0.97,                // 0..1, calibrated
  "probabilities": {"road_damage": 0.97, "garbage": 0.01, "...": 0.0},
  "best_civic_label": "road_damage", // best guess ignoring not_civic
  "is_civic": true,                  // false = clearly irrelevant photo (selfie, food, indoor...)
  "needs_review": false,             // true = low confidence, send to the human review queue
  "embedding": [/* 512 numbers */],  // for duplicate detection
  "model": "ViT-B-32/openai frozen + logistic-regression head"
}
```

Failure (any non-200 status, network error or timeout):
```json
{"error": "...", "fallback": "vlm_only"}
```
**Never block a citizen's report because the classifier failed.** Continue with the VLM alone.

`GET /health` returns `{"ok": true, ...}` when the service is up.

## Recommended flow (do this on the server, e.g. a Next.js API route)

```
browser ── photo + text + location ──▶ our backend /api/report
                                         ├─▶ classifier /classify   (≤ 12 s, optional)
                                         ├─▶ hosted VLM              (main engine)
                                         └─▶ save report + decide review
```

Calling from the backend keeps the logic in one place, lets you combine the result with the VLM, and keeps the URL out of the browser bundle.

### Next.js (App Router): `app/api/report/route.ts`
```ts
export const runtime = "nodejs";

const LABELS = ["road_damage", "garbage", "water_issue", "streetlight", "pollution", "not_civic"];

async function classify(photo: File) {
  try {
    const form = new FormData();
    form.append("file", photo, photo.name || "photo.jpg");
    const r = await fetch(`${process.env.CLASSIFIER_URL}/classify`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(12_000),
    });
    return r.ok ? await r.json() : null;  // null => VLM-only fallback
  } catch {
    return null;
  }
}

export async function POST(req: Request) {
  const data = await req.formData();
  const photo = data.get("photo") as File;
  const clf = await classify(photo);

  // 1. Gate: reject clearly irrelevant photos before paying for a VLM call.
  if (clf && !clf.is_civic) {
    return Response.json({ status: "rejected", reason: "This photo doesn't look like a civic issue." }, { status: 422 });
  }

  // 2. Main engine. If the VLM fails, fall back to the classifier's best civic label.
  let vlm = null;
  try { vlm = await callVLM(photo, data.get("text")); } catch {}
  const category = vlm?.category ?? clf?.best_civic_label ?? "unknown";

  // 3. Cross-check: humans review when unsure or when the two models disagree.
  const needsReview = !clf || !vlm || clf.needs_review || (LABELS.includes(vlm.category) && clf.label !== vlm.category);

  // 4. Save. Store clf.embedding with the report for duplicate checks.
  const report = await saveReport({ category, needsReview, classifier: clf && { ...clf, embedding: undefined }, embedding: clf?.embedding, vlm });
  return Response.json(report);
}
```
`callVLM` and `saveReport` are yours. Make sure the VLM returns one of the six `LABELS` above (`water_issue` = waterlogging + leaks), so the cross-check can compare them.

### Frontend form (any framework)
```ts
const form = new FormData();
form.append("photo", fileInput.files[0]);
form.append("text", description);
form.append("lat", String(lat));
form.append("lng", String(lng));
const res = await fetch("/api/report", { method: "POST", body: form });
const out = await res.json();
if (res.status === 422) showError(out.reason);      // irrelevant photo
else showSuccess(out);                                // show category, "under review" badge if needsReview
```

### Quick option: call the classifier straight from the browser
CORS is enabled, so this works too (for a prototype, or to show "Detected: Pothole (97%)" instantly before submit):
```ts
const form = new FormData();
form.append("file", file);
const r = await fetch(`${process.env.NEXT_PUBLIC_CLASSIFIER_URL}/classify`, { method: "POST", body: form });
const clf = r.ok ? await r.json() : null;
```
Still repeat the call on the server for anything saved, since the browser can't be trusted.

## Duplicate detection
Embeddings are normalised, so similarity is just the dot product:
```ts
const sim = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
// likely duplicate: same category, within ~50 m, and sim(newEmb, oldEmb) > 0.92
```
0.92 is a starting guess. Check it on a few real pairs.

## Gotchas
- The first request after the server (re)starts can take about 1 s; after that it's about 30–300 ms (mostly network).
- The tunnel URL changes when it is restarted, so keep it in an env var and redeploy after changing it.
- On Vercel: Settings → Environment Variables → `CLASSIFIER_URL`, then redeploy.
- Don't show `probabilities` or `embedding` to citizens; show `label` (pretty name) and the review status.
