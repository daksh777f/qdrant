"""Embedder plumbing. The ONNX path is checked with a tiny model built here (no downloads):
it must mean-pool over real tokens only and return unit vectors."""

from __future__ import annotations

import numpy as np
import pytest

from loci.edge.embed import HashEmbedder, make_embedder


def test_hash_stand_in_is_labelled_and_unit_norm():
    e = HashEmbedder(32)
    assert e.real is False and "not a semantic model" in e.name
    v = e.embed_many(["red toolbox", "oil spill"])
    assert v.shape == (2, 32) and np.allclose(np.linalg.norm(v, axis=1), 1)


def test_auto_falls_back_visibly_when_fastembed_is_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_fastembed(name, *a, **k):
        if name == "fastembed":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    e, note = make_embedder("auto")
    assert e.real is False and "fastembed unavailable" in note


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError):
        make_embedder("word2vec")


def _tiny_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tokenizers = pytest.importorskip("tokenizers")
    from onnx import TensorProto, helper, numpy_helper

    vocab = {"[PAD]": 0, "[UNK]": 1, "red": 2, "toolbox": 3, "oil": 4, "spill": 5}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    tok.save(str(tmp_path / "tokenizer.json"))
    # Embedding table: token id -> 3-d vector. last_hidden_state = Gather(table, input_ids).
    table = np.array(
        [[9, 9, 9], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=np.float32
    )
    graph = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"])],
        "tiny",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None]),
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [None, None, 3])],
        [numpy_helper.from_array(table, "table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "model.onnx"))
    return table


def test_onnx_embedder_mean_pools_real_tokens_only(tmp_path):
    table = _tiny_model(tmp_path)
    e, note = make_embedder(f"onnx:{tmp_path}")
    assert e.real and e.dim == 3 and "ONNX" in note
    # "red toolbox" is padded next to the 1-token "oil": padding (row 0 = [9,9,9]) must be ignored
    v = e.embed_many(["red toolbox", "oil"])
    expect = (table[2] + table[3]) / 2
    assert np.allclose(v[0], expect / np.linalg.norm(expect), atol=1e-6)
    assert np.allclose(v[1], table[4] / np.linalg.norm(table[4]), atol=1e-6)
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
