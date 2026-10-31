import numpy as np
import torch
import pytest

from graph_builder import build_frozen_item_graph


def _dense(sp):
    return sp.to_dense()


def test_graph_is_symmetric():
    emb = np.random.RandomState(0).randn(40, 16)
    g = _dense(build_frozen_item_graph(emb, k=5, device=torch.device("cpu")))
    assert torch.allclose(g, g.T, atol=1e-6)


def test_no_self_loops():
    emb = np.random.RandomState(1).randn(40, 16)
    g = _dense(build_frozen_item_graph(emb, k=5, device=torch.device("cpu")))
    assert torch.count_nonzero(torch.diagonal(g)) == 0


def test_no_isolated_items():
    emb = np.random.RandomState(2).randn(40, 16)
    g = _dense(build_frozen_item_graph(emb, k=5, device=torch.device("cpu")))
    assert (g.sum(dim=1) > 0).all()


def test_values_are_symmetric_normalisation():
    """Each non-zero must equal 1/sqrt(d_i * d_j) of the binary symmetrised graph."""
    emb = np.random.RandomState(3).randn(30, 8)
    g = _dense(build_frozen_item_graph(emb, k=4, device=torch.device("cpu")))
    binary = (g > 0).float()
    deg = binary.sum(dim=1)
    expected = binary / torch.sqrt(torch.outer(deg, deg))
    assert torch.allclose(g, expected, atol=1e-6)


def test_exactly_k_neighbours_before_symmetrisation():
    """Every row must have at least k neighbours; symmetrisation only adds."""
    emb = np.random.RandomState(4).randn(50, 12)
    g = _dense(build_frozen_item_graph(emb, k=6, device=torch.device("cpu")))
    assert ((g > 0).sum(dim=1) >= 6).all()


def test_deterministic():
    emb = np.random.RandomState(5).randn(40, 16)
    a = _dense(build_frozen_item_graph(emb, k=5, device=torch.device("cpu")))
    b = _dense(build_frozen_item_graph(emb, k=5, device=torch.device("cpu")))
    assert torch.equal(a, b)


def test_k_clamped_to_n_minus_one():
    emb = np.random.RandomState(6).randn(5, 8)
    g = _dense(build_frozen_item_graph(emb, k=99, device=torch.device("cpu")))
    assert ((g > 0).sum(dim=1) == 4).all()


def test_accepts_float64_input():
    """The pickles on disk are float64; MPS rejects float64, so it must be cast."""
    emb = np.random.RandomState(7).randn(20, 8).astype(np.float64)
    g = build_frozen_item_graph(emb, k=3, device=torch.device("cpu"))
    assert g.dtype == torch.float32


def test_chunking_matches_single_chunk():
    emb = np.random.RandomState(8).randn(64, 16)
    big = _dense(build_frozen_item_graph(emb, k=5, chunk_size=1024, device=torch.device("cpu")))
    small = _dense(build_frozen_item_graph(emb, k=5, chunk_size=7, device=torch.device("cpu")))
    assert torch.allclose(big, small, atol=1e-6)
