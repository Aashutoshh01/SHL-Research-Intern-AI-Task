"""Build FAISS vector store from SHL catalog.

One-time script that:
1. Loads the SHL-Catalogue.txt JSON
2. Creates embedding text for each entry
3. Embeds using sentence-transformers
4. Builds a FAISS index
5. Saves index and metadata to disk

Usage:
    conda activate conv-env
    python -m scripts.build_vectorstore
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.models.catalog import CatalogEntry


def main() -> None:
    """Build the FAISS vector store from the SHL catalog."""
    settings = get_settings()

    # --- Load catalog ---
    catalog_path = settings.absolute_catalog_path
    print(f"Loading catalog from: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        raw_data = json.loads(f.read(), strict=False)

    entries = [CatalogEntry(**item) for item in raw_data]
    print(f"Loaded {len(entries)} catalog entries")

    # --- Generate embedding texts ---
    texts = [entry.embedding_text for entry in entries]
    print(f"Generated {len(texts)} embedding texts")
    print(f"Sample text: {texts[0][:150]}...")

    # --- Load embedding model ---
    print(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    # --- Compute embeddings ---
    print("Computing embeddings (this may take a moment)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    )

    embeddings_np = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings_np.shape}")

    # --- Build FAISS index ---
    dimension = embeddings_np.shape[1]
    print(f"Building FAISS index (dimension={dimension})...")

    # Use IndexFlatIP for cosine similarity (embeddings are normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    print(f"FAISS index built with {index.ntotal} vectors")

    # --- Save index and metadata ---
    index_path = settings.absolute_faiss_index_path
    metadata_path = settings.absolute_faiss_metadata_path

    # Create directories if needed
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to: {index_path}")

    with open(metadata_path, "wb") as f:
        pickle.dump(entries, f)
    print(f"Catalog metadata saved to: {metadata_path}")

    # --- Verification ---
    print("\n--- Verification ---")
    test_query = "Java programming senior engineer"
    query_embedding = model.encode([test_query], normalize_embeddings=True)
    query_vector = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_vector, 5)
    print(f"Test query: '{test_query}'")
    print("Top 5 results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        entry = entries[idx]
        print(f"  {i+1}. {entry.name} (score: {dist:.4f})")

    print("\nVector store build complete!")


if __name__ == "__main__":
    main()
