"""FastAPI service: POST /classify (multipart field "file") -> classifier.predict JSON.

Errors always come back as {"error": ..., "fallback": "vlm_only"} so the web
backend can skip the classifier and use the VLM alone.
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import classifier

TIMEOUT_S = 10
MAX_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await run_in_threadpool(classifier._load)  # load weights once, not on the first request
    yield


app = FastAPI(title="Civic image classifier", lifespan=lifespan)
# Let the website call us straight from the browser (no cookies or credentials involved).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def fail(status, error):
    return JSONResponse(status_code=status, content={"error": error, "fallback": "vlm_only"})


@app.get("/")
def root():
    return {"service": "civic-classifier", "endpoints": ["GET /health", "POST /classify"]}


@app.get("/health")
def health():
    meta = classifier._load()[0]
    return {"ok": True, "encoder": f"{meta['open_clip_arch']}/{meta['open_clip_pretrained']}",
            "classes": meta["classes"]}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    data = await file.read(MAX_BYTES + 1)
    if not data:
        return fail(400, "empty file")
    if len(data) > MAX_BYTES:
        return fail(413, "file larger than 10 MB")
    try:
        return await asyncio.wait_for(run_in_threadpool(classifier.predict, data), TIMEOUT_S)
    except asyncio.TimeoutError:
        return fail(504, f"classifier timed out after {TIMEOUT_S}s")
    except Exception as e:  # unreadable image, model error, ...
        return fail(422, f"could not classify image: {type(e).__name__}: {e}")
