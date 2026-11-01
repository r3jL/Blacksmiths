# Wiring the Lovable app to the classifier

The classifier is a separate Python service (PyTorch). Lovable hosting can't run it,
so the browser talks to it over HTTPS.

**Live URL (Modal, deployed):**
`https://kedia-pradyun--civic-classifier-fastapi-app.us-east.modal.run`

Redeploy after changing `api.py` / `classifier.py` / `artifacts/`:

```bash
cd ~/Desktop/Blacksmiths
uvx modal deploy modal_app.py     # ~90 s; rebuilds only the changed layers
uvx modal app list                # see the app + logs link
```

Redeploying keeps the same URL, so the Lovable app needs no change.

Put the URL in an environment variable, never hard-code it:

- Vite / Lovable default stack: `VITE_CLASSIFIER_URL`
- Next.js: `NEXT_PUBLIC_CLASSIFIER_URL` (only if you accept that the URL is public)

CORS is already `allow_origins=["*"]`, so a direct browser call works.

## 1. The client helper — `src/lib/classifier.ts`

```ts
export const CLASSIFIER_URL = import.meta.env.VITE_CLASSIFIER_URL as string;

export const LABELS = [
  "road_damage", "garbage", "water_issue",
  "streetlight", "pollution", "not_civic",
] as const;
export type Label = (typeof LABELS)[number];

export type ClassifierResult = {
  label: Label;
  probabilities: Record<Label, number>;
  confidence: number;
  best_civic_label: Exclude<Label, "not_civic">;
  is_civic: boolean;
  needs_review: boolean;
  embedding: number[]; // 512 floats, L2-normalised
  model: string;
};

/** Returns null on any failure. Never block a citizen's report on this. */
export async function classifyPhoto(photo: File): Promise<ClassifierResult | null> {
  try {
    const form = new FormData();
    form.append("file", photo, photo.name || "photo.jpg");
    const r = await fetch(`${CLASSIFIER_URL}/classify`, {
      method: "POST",
      body: form,
      // First request after idle wakes the container: allow for a cold start.
      signal: AbortSignal.timeout(60_000),
    });
    if (!r.ok) return null;
    return (await r.json()) as ClassifierResult;
  } catch {
    return null; // network error, timeout, CORS -> VLM-only fallback
  }
}

export function cosine(a: number[], b: number[]) {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}
```

## 2. Use it in the report form

```tsx
const [clf, setClf] = useState<ClassifierResult | null>(null);
const [checking, setChecking] = useState(false);

async function onPick(file: File) {
  setPhoto(file);
  setChecking(true);
  setClf(await classifyPhoto(file));
  setChecking(false);
}

// Gate: refuse clearly irrelevant photos before spending a VLM call.
const blocked = clf !== null && !clf.is_civic;
```

Show the citizen only the pretty label and the review status:

```tsx
{checking && <p className="text-sm text-muted-foreground">Checking photo…</p>}
{clf?.is_civic && (
  <p className="text-sm">
    Detected: <strong>{pretty(clf.best_civic_label)}</strong>{" "}
    ({Math.round(clf.confidence * 100)}%)
    {clf.needs_review && <span className="ml-2 text-amber-600">will be reviewed</span>}
  </p>
)}
{blocked && (
  <p className="text-sm text-destructive">
    This photo doesn't look like a civic issue. Try a photo of the problem itself.
  </p>
)}
```

Disable submit when `blocked`.

**Never render `probabilities` or `embedding` to citizens** — 512 numbers are useless
to them and the probabilities invite arguments.

## 3. Full flow (gate + cross-check + fallback)

Mirrors `INTEGRATION.md` in the backend repo:

```ts
const clf = await classifyPhoto(photo);

// 1. Gate
if (clf && !clf.is_civic) return { status: "rejected" };

// 2. Main engine
let vlm = null;
try { vlm = await callVLM(photo, description); } catch { /* fall through */ }

// 3. Category: VLM first, classifier as fallback
const category = vlm?.category ?? clf?.best_civic_label ?? "unknown";

// 4. Cross-check: unsure, missing, or disagreement -> human review
const needsReview =
  !clf || !vlm || clf.needs_review ||
  (LABELS.includes(vlm.category) && clf.label !== vlm.category);
```

Store `clf.embedding` with the report so duplicates can be found later
(`cosine > 0.92` within ~50 m suggests the same scene; 0.92 is untested on real
reports, tune it).

## 4. Duplicate detection

```ts
const sim = cosine(newEmbedding, existing.embedding);
const duplicate = sim > 0.92 && sameCategory && distanceMeters < 50;
```

## Verified against the live deployment (2026-11-01)

| request | result |
|---|---|
| `GET /` | `{"service":"civic-classifier",...}` |
| `GET /health` | `{"ok":true,"encoder":"ViT-B-32/openai",...}` |
| pothole photo (`road_damage`) | `label=road_damage`, confidence **0.9995**, `is_civic=true`, `needs_review=false`, 2.3 s warm |
| street scene (`not_civic`) | `label=not_civic`, p=**0.577** → `is_civic=true`, `needs_review=true` |

That second row is the gate working as designed: `not_civic` only *rejects* at
p ≥ 0.8, so a borderline image is passed to the VLM and flagged for review instead
of being thrown away. Expect this often — on the 104-photo test set the gate caught
only 15 of 37 irrelevant photos at that threshold.

## Gotchas

- **Cold start:** first request after ~5 idle minutes takes 10–60 s. The client
  timeout above is 60 s. Optionally call `GET /health` when the form mounts to warm it.
- **10 MB limit:** server rejects larger uploads with 413. Downscale in the browser
  before upload if you expect big phone photos.
- **Never block a report on the classifier.** A null result means "ask the VLM only".
- The service returns `{"error": ..., "fallback": "vlm_only"}` on failure — treat
  every non-200 as "classifier unavailable".
