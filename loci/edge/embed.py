"""Text embedders for the edge: a real model when one is available, a labelled stand-in otherwise.

Every embedder has ``name``, ``dim``, ``real`` (is it a learned semantic model?) and is a
``text -> list[float]`` callable returning unit vectors. :func:`make_embedder` picks one from a
spec string (or ``LOCI_EMBEDDER``):

* ``"fastembed[:MODEL]"``: Qdrant's own ``fastembed`` library (ONNX, CPU, no PyTorch). Downloads
  the model on first use. Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (384-d).
* ``"onnx:/path/to/model_dir"``: a pre-provisioned sentence-transformer ONNX export
  (``model.onnx`` or ``onnx/model.onnx`` + ``tokenizer.json``), mean-pooled and normalised. For
  robots that are provisioned once and never download anything.
* ``"hash[:DIM]"``: hashed bag-of-words. Not a semantic model: texts match only through shared
  words. It exists so the demo and tests run with zero downloads, and it is always labelled.

``"auto"`` (the default) tries fastembed, then falls back to the stand-in and says so.
"""

from __future__ import annotations

import os
import re
import zlib
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str
    dim: int
    real: bool

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> np.ndarray: ...

    def __call__(self, text: str) -> list[float]: ...


class _Base:
    name = "base"
    dim = 0
    real = False

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def __call__(self, text: str) -> list[float]:
        return cast("list[float]", self.embed(text).tolist())


class HashEmbedder(_Base):
    """Hashed bag-of-words stand-in (see module docstring). Deterministic, zero downloads."""

    real = False

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.name = f"hash-bow-{dim} (stand-in, not a semantic model)"
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

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim))


class FastEmbedEmbedder(_Base):
    """Qdrant's ``fastembed`` (``pip install fastembed``)."""

    real = True

    def __init__(self, model: str = DEFAULT_FASTEMBED_MODEL, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self.dim = int(self._model.embedding_size)
        self.name = f"fastembed:{model}"

    def embed_many(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class OnnxEmbedder(_Base):
    """A sentence-transformer ONNX export on local disk (mean pooling + L2 normalisation)."""

    real = True

    def __init__(self, model_dir: str | Path, max_length: int = 256, batch: int = 64) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        onnx_file = next(
            (p for p in (d / "model.onnx", d / "onnx" / "model.onnx") if p.exists()), None
        )
        if onnx_file is None or not (d / "tokenizer.json").exists():
            raise FileNotFoundError(f"{d} needs model.onnx (or onnx/model.onnx) and tokenizer.json")
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_length)
        self._tok.enable_padding()
        self._sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._batch = batch
        self.dim = int(self.embed_many(["probe"]).shape[1])
        self.name = f"onnx:{d.name}"

    def embed_many(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self._batch):
            enc = self._tok.encode_batch(texts[i : i + self._batch])
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feed: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            if hidden.ndim == 2:  # already pooled
                pooled = hidden
            else:
                m = mask[..., None].astype(np.float32)
                pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


def make_embedder(spec: str | None = None, *, hash_dim: int = 64) -> tuple[Embedder, str]:
    """Build an embedder from *spec* (or ``LOCI_EMBEDDER``). Returns ``(embedder, note)``."""
    spec = spec or os.environ.get("LOCI_EMBEDDER", "auto")
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return HashEmbedder(int(arg) if arg else hash_dim), "hash stand-in requested"
    if kind == "onnx":
        return OnnxEmbedder(arg), f"local ONNX model at {arg}"
    if kind == "fastembed":
        return FastEmbedEmbedder(arg or DEFAULT_FASTEMBED_MODEL), "fastembed"
    if kind == "auto":
        try:
            return FastEmbedEmbedder(), "fastembed"
        except Exception as exc:  # not installed, or the model could not be downloaded
            reason = f"{type(exc).__name__}: {str(exc)[:120]}"
            return HashEmbedder(hash_dim), f"fastembed unavailable ({reason}); using hash stand-in"
    raise ValueError(f"unknown embedder spec {spec!r}")
