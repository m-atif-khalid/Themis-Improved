"""BM25 + Levenshtein-automata style search for database text fields.

This module combines:
- BM25 scoring (information retrieval)
- RapidFuzz for Levenshtein-based fuzzy matching

It returns top-N suggestions for a query across a chosen table/column.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Tuple

import mysql.connector
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def _fetch_column_values(
    db_config: Dict[str, Any],
    table: str,
    column: str,
) -> List[str]:
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    try:
        cursor.execute(f"SELECT {column} FROM {table}")
        return [row[0] for row in cursor.fetchall() if row and row[0] is not None]
    finally:
        cursor.close()
        connection.close()


def _bm25_scores(query: str, corpus: Iterable[str]) -> List[Tuple[str, float]]:
    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    return list(zip(corpus, scores))


def _fuzzy_scores(query: str, corpus: Iterable[str]) -> List[Tuple[str, float]]:
    # RapidFuzz returns scores in [0, 100]
    results = process.extract(
        query,
        corpus,
        scorer=fuzz.WRatio,
        limit=None,
    )
    return [(match, score) for match, score, _idx in results]


def search_suggestions(
    query: str,
    corpus: List[str],
    top_n: int = 7,
) -> List[Tuple[str, float]]:
    """Return top-N suggestions combining BM25 + fuzzy matching scores."""
    if not corpus:
        return []

    bm25 = dict(_bm25_scores(query, corpus))
    fuzzy = dict(_fuzzy_scores(query, corpus))

    combined = []
    for item in corpus:
        bm25_score = bm25.get(item, 0.0)
        fuzzy_score = fuzzy.get(item, 0.0) / 100.0  # normalize to [0, 1]
        # Weighted combination: prioritize fuzzy for near matches, bm25 for IR relevance
        combined_score = (0.6 * fuzzy_score) + (0.4 * bm25_score)
        combined.append((item, combined_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_n]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BM25 + Levenshtein fuzzy search")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--table", required=True, help="DB table to search")
    parser.add_argument("--column", required=True, help="DB column to search")
    parser.add_argument("--top", type=int, default=7, help="Number of suggestions")
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

    values = _fetch_column_values(db_config, args.table, args.column)
    suggestions = search_suggestions(args.query, values, args.top)

    if not suggestions:
        print("No suggestions found.")
        return

    print("Top suggestions:")
    for value, score in suggestions:
        print(f"- {value} (score: {score:.4f})")


if __name__ == "__main__":
    main()
