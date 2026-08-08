Malicious URL & Phishing Classifier
====================================

This repository contains a small, runnable skeleton of the "Malicious URL & Phishing Classifier" project.

Quickstart (Windows PowerShell):

1. Open this folder in VS Code.
2. Create and activate virtualenv:
   py -3 -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
3. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
4. Run tests:
   pytest -q
5. Demo feature extraction:
   python -m src.data.ingest --input data\raw\kaggle_dataset.csv --output-dir   data
6. Train a small demo model:
   python -m src.models.train --data data\\train.csv --output-dir src\\models\\artifacts
7. Run FastAPI (if model artifact present):
   uvicorn src.api.main:app --reload --port 8000

Files added by this scaffold:
- src/features/feature_extractor.py  (lexical feature extractor)
- src/data/ingest.py                (reads CSV of urls, extracts features)
- src/models/train.py               (train a demo RandomForest on synthetic / processed data)
- src/api/main.py                   (FastAPI inference endpoint)
- demo_predict.py                   (simple CLI to call a saved model)
- tests/test_feature_extractor.py   (unit test for feature extractor)
- .github/workflows/ci.yml          (CI: install deps + run pytest)

