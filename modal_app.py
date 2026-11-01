"""Deploy the civic classifier to Modal as a public HTTPS endpoint (free Starter plan).

    uvx modal token new          # one-time browser login
    uvx modal deploy modal_app.py

Prints a stable URL like:
    https://<workspace>--civic-classifier-fastapi-app.modal.run

The container image is built in Modal's cloud, so nothing heavy installs locally.
CLIP ViT-B/32 weights (~577 MB) are downloaded during the image build, not on the
first request, which keeps cold starts to a few seconds.

Endpoints are the ones api.py already serves: GET /health, POST /classify.
"""
import os
import sys

import modal

APP_DIR = "/root/app"

app = modal.App("civic-classifier")

# Torch comes from the CPU wheel index: the default Linux wheel drags in ~3 GB of
# CUDA libraries we would never use on a CPU container.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "open-clip-torch==3.3.0",
        "timm==1.0.30",
        "huggingface-hub==2.4.0",
        "scikit-learn==1.9.1",
        "numpy==2.5.4",
        "joblib==1.6.0",
        "pillow==13.0.0",
        "fastapi==0.162.0",
        "python-multipart==0.0.33",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTHONPATH": APP_DIR})
    # Only inference needs these; the training data and scripts stay out of the image.
    .add_local_file("api.py", f"{APP_DIR}/api.py", copy=True)
    .add_local_file("classifier.py", f"{APP_DIR}/classifier.py", copy=True)
    .add_local_file("common.py", f"{APP_DIR}/common.py", copy=True)
    .add_local_dir("artifacts", f"{APP_DIR}/artifacts", copy=True)
)


def _bake_weights():
    """Run at image build time so the CLIP weights land in the image cache."""
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    import classifier

    classifier._load()
    print("CLIP weights cached in image")


image = image.run_function(_bake_weights)


@app.function(
    image=image,
    cpu=2.0,
    memory=2048,
    timeout=120,
    # Keep one container warm for 5 minutes after the last request. Longer windows
    # burn the $30/month credit faster; the client should tolerate a cold start.
    scaledown_window=300,
    max_containers=2,
)
@modal.asgi_app()
def fastapi_app():
    from api import app as api_app

    return api_app
