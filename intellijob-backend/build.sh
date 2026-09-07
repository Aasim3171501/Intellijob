#!/usr/bin/env bash
set -o errexit

# Render build script - runs on every deploy
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model (cached after first build)
python -m spacy download en_core_web_sm

# Collect static files (if any)
python manage.py collectstatic --noinput

echo "Build complete"