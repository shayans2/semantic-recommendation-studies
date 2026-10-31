import numpy as np
import scipy.sparse as sp
import torch

from utils import degree_sensitive_prune


def _mat(seed=0, n_users=60, n_items=40, density=0.08):
    rng = np.random.RandomState(seed)
    dense = (rng.rand(n_users, n_items) < density).astype(np.float32)
    dense[np.arange(n_users), rng.randint(0, n_items, n_users)] = 1.0  # no empty user
    return sp.coo_matrix(dense)


def test_keeps_expected_edge_count():
    m = _mat()
    g = torch.Generator().manual_seed(0)
    out = degree_sensitive_prune(m, rho=0.2, device=torch.device("cpu"), generator=g)
    # bipartite adjacency stores each kept interaction twice (u->i and i->u)
    assert out._nnz() == 2 * round(0.8 * m.nnz)


def test_output_is_symmetric():
    m = _mat(1)
    g = torch.Generator().manual_seed(1)
    out = degree_sensitive_prune(m, rho=0.3, device=torch.device("cpu"), generator=g).to_dense()
    assert torch.allclose(out, out.T, atol=1e-6)


def test_renormalised_on_pruned_degrees():
    """Values must be 1/sqrt(d_i*d_j) of the PRUNED graph, not the full one."""
    m = _mat(2)
    g = torch.Generator().manual_seed(2)
    out = degree_sensitive_prune(m, rho=0.4, device=torch.device("cpu"), generator=g).to_dense()
    binary = (out > 0).float()
    deg = binary.sum(dim=1)
    expected = binary / torch.sqrt(torch.outer(deg, deg))
    expected[torch.isnan(expected) | torch.isinf(expected)] = 0.0
    assert torch.allclose(out, expected, atol=1e-6)


def test_drops_hub_edges_more_than_tail_edges():
    """The direction fixed in spec 2.2: hubs must be pruned MORE than the tail."""
    m = _mat(3, n_users=200, n_items=120, density=0.05)
    wu = np.asarray((m != 0).sum(1)).ravel()
    wi = np.asarray((m != 0).sum(0)).ravel()
    deg_pair = wu[m.row] * wi[m.col]
    lo = deg_pair <= np.quantile(deg_pair, 0.25)
    hi = deg_pair >= np.quantile(deg_pair, 0.75)

    kept = np.zeros(m.nnz)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        out = degree_sensitive_prune(m, rho=0.3, device=torch.device("cpu"), generator=g).to_dense()
        kept += np.array([out[r, m.shape[0] + c].item() > 0 for r, c in zip(m.row, m.col)])
    drop_rate_tail = 1 - kept[lo].mean() / 20
    drop_rate_hub = 1 - kept[hi].mean() / 20
    assert drop_rate_hub > drop_rate_tail


def test_reproducible_with_same_generator():
    m = _mat(4)
    a = degree_sensitive_prune(m, 0.25, torch.device("cpu"), torch.Generator().manual_seed(7)).to_dense()
    b = degree_sensitive_prune(m, 0.25, torch.device("cpu"), torch.Generator().manual_seed(7)).to_dense()
    assert torch.equal(a, b)


def test_rho_zero_keeps_every_edge():
    m = _mat(5)
    g = torch.Generator().manual_seed(5)
    out = degree_sensitive_prune(m, rho=0.0, device=torch.device("cpu"), generator=g)
    assert out._nnz() == 2 * m.nnz
