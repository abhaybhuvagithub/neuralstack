"""ingest.py — load source documents and split them into retrievable chunks.

Chunking strategy: fixed-size character windows with overlap. The overlap
keeps sentences that straddle a boundary retrievable from both chunks,
which noticeably reduces "answer was split across two chunks" misses.

A production system would chunk on token counts (using the model's
tokenizer) and prefer semantic boundaries (paragraphs, headings), but the
window-with-overlap approach is the dependable baseline.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
    source: str = "unknown",
) -> List[Tuple[str, Dict[str, Any]]]:
    """Split `text` into overlapping windows, tagged with source metadata."""
    text = " ".join(text.split())  # normalise whitespace
    chunks: List[Tuple[str, Dict[str, Any]]] = []
    start, idx = 0, 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        window = text[start : start + chunk_size]
        chunks.append((window, {"source": source, "chunk": idx}))
        start += step
        idx += 1
    return chunks


def load_documents(folder: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Read every .txt/.md file in `folder` and return all their chunks."""
    all_chunks: List[Tuple[str, Dict[str, Any]]] = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith((".txt", ".md")):
            continue
        path = os.path.join(folder, name)
        with open(path, "r", encoding="utf-8") as fh:
            all_chunks.extend(chunk_text(fh.read(), source=name))
    return all_chunks
