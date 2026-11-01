#!/bin/bash
# Rebuild everything after adding photos to data/own/<class>/ or data/demo/<class>/.
set -e
cd "$(dirname "$0")"
PY=venv/bin/python
$PY check_data.py
$PY embed.py clip_b32
$PY split.py
$PY compare.py clip_b32
$PY train.py --encoder clip_b32
$PY evaluate.py
