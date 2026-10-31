import numpy as np
import torch as t
import torch.nn.functional as F

from utils import get_device


def build_frozen_item_graph(item_sem_emb, k=10, chunk_size=2048, device=None):
    """Phase 2: a frozen, symmetrically normalised item-item graph.

    Cosine similarity over the LLM item embeddings, sparsified to the top-k
    neighbours per item, discretised to 1/0, symmetrised, then normalised as
    D^-1/2 S D^-1/2.

    The graph is symmetrised BEFORE normalising. A binary top-k graph has
    exactly k entries per row, so D = kI and the normalisation would otherwise
    reduce to a constant 1/k rescale rather than a degree-aware filter.

    Args:
        item_sem_emb: (n_items, d) array or tensor of item semantic embeddings.
        k: neighbours retained per item, clamped to n_items - 1.
        chunk_size: rows per similarity chunk. The full similarity matrix is
            O(n_items^2) and is never materialised.
        device: target device; defaults to utils.get_device().

    Returns:
        Coalesced sparse COO float32 tensor of shape (n_items, n_items).
    """
    device = get_device() if device is None else device

    x = t.as_tensor(np.asarray(item_sem_emb)).float()
    n_items = x.shape[0]
    k = max(1, min(k, n_items - 1))
    x = F.normalize(x, p=2, dim=1).to(device)

    rows, cols = [], []
    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        sim = x[start:stop] @ x.T
        # Exclude self-similarity so an item is never its own neighbour.
        sim[t.arange(stop - start, device=device), t.arange(start, stop, device=device)] = float("-inf")
        nbr = sim.topk(k, dim=1).indices
        rows.append(t.arange(start, stop, device=device).repeat_interleave(k))
        cols.append(nbr.reshape(-1))

    rows = t.cat(rows)
    cols = t.cat(cols)

    # Symmetrise as a binary union: an edge exists if either endpoint chose the
    # other. Concatenating both directions then clamping values to 1 does this.
    sym_rows = t.cat([rows, cols])
    sym_cols = t.cat([cols, rows])
    idxs = t.stack([sym_rows, sym_cols])
    adj = t.sparse_coo_tensor(
        idxs, t.ones(idxs.shape[1], device=device), (n_items, n_items)
    ).coalesce()
    idxs = adj.indices()
    vals = t.ones(idxs.shape[1], device=device)  # re-binarise after coalesce summed duplicates

    deg = t.zeros(n_items, device=device).index_add_(0, idxs[0], vals)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[t.isinf(d_inv_sqrt)] = 0.0
    vals = d_inv_sqrt[idxs[0]] * vals * d_inv_sqrt[idxs[1]]

    return t.sparse_coo_tensor(idxs, vals, (n_items, n_items)).coalesce()


def build_random_item_graph(n_items, k=10, device=None, generator=None):
    """Ablation control: a random item-item graph, structurally matched.

    Selects k random neighbours per item instead of the k most semantically
    similar, then applies exactly the same symmetrisation and normalisation as
    build_frozen_item_graph. Item count, k, edge count, symmetry, degree
    normalisation and the propagation path are all held constant, so the ONLY
    thing that differs is whether the topology carries semantic information.

    This is the control for the central claim that the LLM embeddings are what
    make the frozen graph useful. If a model performs the same on this graph,
    the semantics are not doing the work -- merely having a graph is.

    Draws from the global torch RNG unless a generator is supplied, so a seeded
    run gets a reproducible graph and different seeds get different ones.
    """
    device = get_device() if device is None else device
    k = max(1, min(k, n_items - 1))

    rows = t.arange(n_items).repeat_interleave(k)
    cols = t.randint(0, n_items - 1, (n_items * k,), generator=generator)
    # Shift past the diagonal so an item is never its own neighbour, matching
    # build_frozen_item_graph's exclusion of self-similarity.
    cols = cols + (cols >= rows).long()

    sym_rows = t.cat([rows, cols]).to(device)
    sym_cols = t.cat([cols, rows]).to(device)
    idxs = t.stack([sym_rows, sym_cols])
    adj = t.sparse_coo_tensor(
        idxs, t.ones(idxs.shape[1], device=device), (n_items, n_items)
    ).coalesce()
    idxs = adj.indices()
    vals = t.ones(idxs.shape[1], device=device)

    deg = t.zeros(n_items, device=device).index_add_(0, idxs[0], vals)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[t.isinf(d_inv_sqrt)] = 0.0
    vals = d_inv_sqrt[idxs[0]] * vals * d_inv_sqrt[idxs[1]]

    return t.sparse_coo_tensor(idxs, vals, (n_items, n_items)).coalesce()


def build_permuted_item_graph(item_sem_emb, k=10, chunk_size=2048, device=None,
                              generator=None):
    """Ablation control: the real semantic graph with node labels shuffled.

    Builds the genuine kNN graph, then relabels items by a random permutation.
    The result is ISOMORPHIC to the real graph -- identical edge count, identical
    degree sequence, identical clustering -- but item i now inherits the
    neighbourhood of some unrelated item, so the topology carries no information
    about item i's semantics.

    This is a stricter control than build_random_item_graph, which also changes
    the degree distribution (a uniform random graph has far less neighbour
    overlap, so symmetrisation nearly doubles its edges while the semantic
    graph's merge). Holding degree structure fixed means a performance drop here
    can only be attributed to the loss of semantic correspondence.
    """
    device = get_device() if device is None else device
    g = build_frozen_item_graph(item_sem_emb, k=k, chunk_size=chunk_size,
                                device=device)
    n_items = g.shape[0]
    perm = t.randperm(n_items, generator=generator).to(device)
    idxs = g.indices()
    return t.sparse_coo_tensor(
        perm[idxs], g.values(), (n_items, n_items)
    ).coalesce()
