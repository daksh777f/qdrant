"""A tiny deterministic text embedder for demos and tests.

Hashed bag-of-words: every token maps to a fixed random unit vector and a text
is the normalised sum, so texts that share words are close and unrelated texts
are near-orthogonal. It needs no model download and works offline, which suits
the demo. It is a stand-in: swap in a real model (e.g. fastembed / bge-small)
by passing any ``text -> list[float]`` callable where an embedder is expected.
"""

from __future__ import annotations

import re
import zlib
from typing import cast

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _token(self, tok: str) -> np.ndarray:
        v = self._cache.get(tok)
        if v is None:
            v = np.random.default_rng(zlib.crc32(tok.encode())).normal(size=self.dim)
            v /= np.linalg.norm(v)
            self._cache[tok] = v
        return v

    def embed(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower()) or [text.lower() or "empty"]
        v = np.sum([self._token(t) for t in toks], axis=0)
        return cast("np.ndarray", v / np.linalg.norm(v))

    def __call__(self, text: str) -> list[float]:
        return cast("list[float]", self.embed(text).tolist())
