# Backend Module

This directory contains the Python backend module(s) for Themis-Improved.

## Structure
- `deepface_module.py`: Image matching + MySQL record lookup using DeepFace.
- `search_module.py`: Text search with BM25 + Levenshtein-based fuzzy matching.
- `likelihood_module.py`: Likelihood + time-to-event prediction with XGBoost, lifelines, and SHAP.
- `csv_data/`: Shared CSV folder for model training and analytics.
- `mysql_data/`: Reserved for local MySQL-related artifacts (no sensitive data committed).
- `images_data/`: Image gallery for matching (filenames should be the person name or primary key ID).

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install deepface mysql-connector-python pandas rank_bm25 rapidfuzz xgboost lifelines shap numpy
```

## Usage
```bash
python backend/deepface_module.py --input path/to/query.jpg --images backend/images_data

python backend/search_module.py --query "john doe" --table criminals --column name --top 7

python backend/likelihood_module.py --world-state Chaotic --db-host localhost --db-user root --db-password "" --db-name themis
```
