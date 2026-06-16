"""embeddings.py — turn text into vectors.

This ships a dependency-free `HashingEmbedder` so the whole project runs
offline. It uses the *feature hashing* trick: each token is hashed into one
of `dim` buckets and accumulated, then the vector is L2-normalised. That
captures lexical overlap well enough to demonstrate retrieval end to end.

For production-grade *semantic* retrieval, swap this class for a real
embedding model behind the same `.embed()` interface — e.g. Voyage AI
(Anthropic's recommended embeddings partner) or a local sentence-transformer.
The rest of the system (vector store, agent) does not change.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def embed(self, text: str) -> List[float]:
        """Map text to a unit-length vector of length `self.dim`."""
        vec = [0.0] * self.dim
        for tok in self._tokenize(text):
            # Stable hash -> bucket index. (Python's built-in hash() is
            # salted per-process, so we use md5 for reproducibility.)
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        # L2 normalise so cosine similarity reduces to a dot product.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]
