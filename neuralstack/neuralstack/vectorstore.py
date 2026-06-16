"""vectorstore.py — a tiny in-memory vector database.

Stores (chunk_text, metadata, vector) records and answers top-k cosine
similarity queries. Because embeddings are L2-normalised, cosine similarity
is just the dot product. For millions of vectors you would replace the
linear scan with an ANN index (FAISS, hnswlib) or a managed store
(Chroma, Pinecone) — but the interface below is all the agent depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .embeddings import HashingEmbedder


@dataclass
class Record:
    text: str
    metadata: Dict[str, Any]
    vector: List[float]


@dataclass
class VectorStore:
    embedder: HashingEmbedder
    records: List[Record] = field(default_factory=list)

    def add(self, text: str, metadata: Dict[str, Any] | None = None) -> None:
        self.records.append(
            Record(text=text, metadata=metadata or {}, vector=self.embedder.embed(text))
        )

    def add_many(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> None:
        for text, meta in chunks:
            self.add(text, meta)

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def search(self, query: str, k: int = 3) -> List[Tuple[Record, float]]:
        """Return the k most similar records as (record, score) pairs."""
        if not self.records:
            return []
        q = self.embedder.embed(query)
        scored = [(r, self._dot(q, r.vector)) for r in self.records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.records)
