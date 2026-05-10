"""DeepFace-based image matching + MySQL record lookup.

Assumptions:
- Gallery images live in `images_data/`.
- Each gallery image filename is either the person's name or a numeric primary key ID.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import mysql.connector
from deepface import DeepFace


def _parse_identity_to_lookup(identity_path: str) -> Tuple[Optional[int], Optional[str]]:
    """Return (person_id, person_name) parsed from a matched image path."""
    base = os.path.basename(identity_path)
    name, _ext = os.path.splitext(base)
    if name.isdigit():
        return int(name), None
    return None, name


def find_best_match(
    input_image_path: str,
    images_dir: str,
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = False,
) -> Optional[str]:
    """Find the best matching image in the gallery.

    Returns the matched gallery image path, or None if no match is found.
    """
    results = DeepFace.find(
        img_path=input_image_path,
        db_path=images_dir,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
    )

    if not results or len(results) == 0:
        return None

    # DeepFace.find returns a list of pandas DataFrames (one per model).
    df = results[0]
    if df.empty:
        return None

    # First row is best match.
    identity_path = df.iloc[0]["identity"]
    return identity_path


def fetch_person_record(
    db_config: Dict[str, Any],
    person_id: Optional[int],
    person_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Fetch a person record from MySQL using ID or name."""
    if person_id is None and not person_name:
        return None

    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    try:
        if person_id is not None:
            cursor.execute("SELECT * FROM criminals WHERE id = %s LIMIT 1", (person_id,))
        else:
            cursor.execute("SELECT * FROM criminals WHERE name = %s LIMIT 1", (person_name,))
        return cursor.fetchone()
    finally:
        cursor.close()
        connection.close()


def match_and_lookup(
    input_image_path: str,
    images_dir: str,
    db_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Match a face image and fetch the corresponding DB record."""
    matched_identity = find_best_match(input_image_path, images_dir)
    if not matched_identity:
        return None

    person_id, person_name = _parse_identity_to_lookup(matched_identity)
    return fetch_person_record(db_config, person_id, person_name)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepFace match + MySQL lookup")
    parser.add_argument("--input", required=True, help="Path to the query image")
    parser.add_argument(
        "--images",
        required=True,
        help="Directory with gallery images named by person name or primary key",
    )
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-name", default="themis")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    db_config = {
        "host": args.db_host,
        "user": args.db_user,
        "password": args.db_password,
        "database": args.db_name,
    }

    record = match_and_lookup(args.input, args.images, db_config)
    if record:
        print("Matched record:")
        print(record)
    else:
        print("No match found.")


if __name__ == "__main__":
    main()
