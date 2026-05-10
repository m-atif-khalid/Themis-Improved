# Backend Module

This directory contains the Python backend module(s) for Themis-Improved.

## Structure
- `deepface_module.py`: Image matching + MySQL record lookup using DeepFace.
- `mysql_data/`: Reserved for local MySQL-related artifacts (no sensitive data committed).
- `images_data/`: Image gallery for matching (filenames should be the person name or primary key ID).

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install deepface mysql-connector-python pandas
```

## Usage
```bash
python backend/deepface_module.py --input path/to/query.jpg --images backend/images_data
```
