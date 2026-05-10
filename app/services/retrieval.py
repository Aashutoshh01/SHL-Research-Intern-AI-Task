"""FAISS vector retrieval service with hybrid lexical re-scoring.

Loads the pre-built FAISS index and performs semantic search
against the SHL catalog. Re-scores results with lexical overlap
to reduce semantic noise. Returns candidate CatalogEntry objects
for downstream filtering and ranking.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.catalog import CatalogEntry

logger = get_logger(__name__)

# Module-level singletons — loaded once, reused across requests
_index: faiss.Index | None = None
_catalog_entries: list[CatalogEntry] = []
_embedding_model: SentenceTransformer | None = None

# Hybrid retrieval blend weights
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45


def _load_catalog_from_json(catalog_path: Path) -> list[CatalogEntry]:
    """Load catalog entries from the raw JSON file.

    Args:
        catalog_path: Path to the SHL-Catalogue.txt JSON file.

    Returns:
        List of validated CatalogEntry objects.
    """
    with open(catalog_path, "r", encoding="utf-8") as f:
        raw_data = json.loads(f.read(), strict=False)
    entries = [CatalogEntry(**item) for item in raw_data]
    logger.info("catalog_loaded", entry_count=len(entries))
    return entries


def initialize_retrieval() -> None:
    """Load FAISS index, catalog metadata, and embedding model.

    This should be called once at application startup.
    Populates module-level singletons for fast retrieval.
    """
    global _index, _catalog_entries, _embedding_model

    settings = get_settings()
    index_path = settings.absolute_faiss_index_path
    metadata_path = settings.absolute_faiss_metadata_path

    # Load embedding model
    logger.info("loading_embedding_model", model=settings.embedding_model)
    _embedding_model = SentenceTransformer(settings.embedding_model)

    # Load FAISS index
    if index_path.exists() and metadata_path.exists():
        logger.info("loading_faiss_index", path=str(index_path))
        _index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            _catalog_entries = pickle.load(f)
        logger.info(
            "faiss_index_loaded",
            vectors=_index.ntotal,
            entries=len(_catalog_entries),
        )
    else:
        logger.warning(
            "faiss_index_not_found",
            index_path=str(index_path),
            message="Run scripts/build_vectorstore.py first",
        )
        # Fallback: load catalog directly for keyword-based retrieval
        _catalog_entries = _load_catalog_from_json(settings.absolute_catalog_path)


def retrieve(query: str, top_k: int | None = None) -> list[tuple[CatalogEntry, float]]:
    """Perform hybrid semantic + lexical retrieval against the catalog.

    Uses FAISS for initial vector retrieval, then re-scores with
    lexical overlap to reduce semantic noise. Falls back to keyword
    matching if the FAISS index isn't available.

    Args:
        query: Search query built from conversation state.
        top_k: Maximum number of results to return.

    Returns:
        List of (CatalogEntry, blended_score) tuples, sorted by relevance.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.retrieval_top_k

    if _index is None or _embedding_model is None:
        logger.warning("faiss_not_available, falling_back_to_keyword_search")
        return _keyword_fallback(query, top_k)

    # Encode query
    query_embedding = _embedding_model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_embedding, dtype=np.float32)

    # Search FAISS index — pull extra candidates for re-ranking
    faiss_k = min(top_k * 2, _index.ntotal)
    distances, indices = _index.search(query_vector, faiss_k)

    # Tokenize query for lexical scoring
    query_tokens = _tokenize(query)

    results: list[tuple[CatalogEntry, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_catalog_entries):
            continue
        entry = _catalog_entries[idx]

        # Semantic similarity (0-1)
        semantic_score = float(dist)  # Already cosine similarity for L2-normalized vectors

        # Lexical overlap score (0-1)
        lexical_score = _lexical_score(query_tokens, entry)

        # Blended score
        blended = SEMANTIC_WEIGHT * semantic_score + LEXICAL_WEIGHT * lexical_score
        results.append((entry, blended))

    # Re-sort by blended score
    results.sort(key=lambda x: x[1], reverse=True)

    # Trim to top_k
    results = results[:top_k]

    logger.info(
        "retrieval_complete",
        query_preview=query[:80],
        results_count=len(results),
    )
    return results


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase terms for lexical matching.

    Splits on non-alphanumeric characters, lowercases, and
    filters out very short tokens.

    Args:
        text: Input text.

    Returns:
        Set of lowercase tokens.
    """
    tokens = re.split(r"[^a-zA-Z0-9#+.]+", text.lower())
    return {t for t in tokens if len(t) >= 2}


def _lexical_score(query_tokens: set[str], entry: CatalogEntry) -> float:
    """Compute lexical overlap between query tokens and a catalog entry.

    Gives 3x weight for matches in the assessment NAME vs. description,
    because name matches are much stronger relevance signals.

    Args:
        query_tokens: Set of lowercase query terms.
        entry: Catalog entry to score.

    Returns:
        Score between 0 and 1.
    """
    if not query_tokens:
        return 0.0

    name_tokens = _tokenize(entry.name)
    desc_tokens = _tokenize(entry.description)

    # Name matches are worth 3x
    name_matches = len(query_tokens & name_tokens)
    desc_matches = len(query_tokens & (desc_tokens - name_tokens))  # Only new desc matches

    # Weighted match count
    weighted_matches = (name_matches * 3.0) + (desc_matches * 1.0)
    max_possible = len(query_tokens) * 3.0  # If all matched in name

    return min(weighted_matches / max_possible, 1.0) if max_possible > 0 else 0.0


def _keyword_fallback(
    query: str, top_k: int
) -> list[tuple[CatalogEntry, float]]:
    """Keyword-based fallback when FAISS index is unavailable.

    Scores each catalog entry by counting query term matches
    in the entry's name and description.

    Args:
        query: Search query string.
        top_k: Maximum results to return.

    Returns:
        List of (CatalogEntry, score) tuples.
    """
    query_terms = set(query.lower().split())
    scored: list[tuple[CatalogEntry, float]] = []

    for entry in _catalog_entries:
        searchable = f"{entry.name} {entry.description}".lower()
        matches = sum(1 for term in query_terms if term in searchable)
        if matches > 0:
            score = matches / len(query_terms) if query_terms else 0.0
            scored.append((entry, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def get_all_catalog_entries() -> list[CatalogEntry]:
    """Return all loaded catalog entries.

    Useful for comparison lookups where we need to find
    specific assessments by name.

    Returns:
        Complete list of CatalogEntry objects.
    """
    return list(_catalog_entries)


def find_entry_by_name(name: str) -> CatalogEntry | None:
    """Find a catalog entry by exact or partial name match.

    Args:
        name: Assessment name to search for.

    Returns:
        Matching CatalogEntry, or None if not found.
    """
    name_lower = name.lower().strip()
    # Exact match first
    for entry in _catalog_entries:
        if entry.name.lower() == name_lower:
            return entry
    # Partial match
    for entry in _catalog_entries:
        if name_lower in entry.name.lower():
            return entry
    return None
