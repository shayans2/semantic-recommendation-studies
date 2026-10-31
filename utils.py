import numpy as np
import torch as t
from torch import nn


def get_device():
    """Pick the best available backend: CUDA, then Apple Metal (MPS), then CPU."""
    if t.cuda.is_available():
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


def spmm(adj, embeds):
    """Sparse-dense matrix product that also works on Apple Metal.

    MPS has no kernel for sparse `aten::mm`, so on that backend the product is
    computed as an explicit gather / scatter-add over the edge list. That is
    algebraically identical to t.spmm and keeps memory at O(nnz * dim).
    """
    if adj.device.type != "mps":
        return t.spmm(adj, embeds)
    idxs = adj._indices()
    vals = adj._values()
    out = t.zeros(
        adj.shape[0], embeds.shape[1], dtype=embeds.dtype, device=embeds.device
    )
    return out.index_add_(0, idxs[0], embeds[idxs[1]] * vals.unsqueeze(1))


class SpAdjEdgeDrop(nn.Module):
    """Bernoulli edge dropout.

    `rescale` divides surviving edges by keep_rate so the expected propagation
    matches the full graph, which is the standard DropEdge convention. RLMRec's
    released code has that division commented out, so `rescale=False` is what a
    faithful reproduction of it needs.
    """

    def __init__(self, rescale=True):
        super(SpAdjEdgeDrop, self).__init__()
        self.rescale = rescale

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        # Build the mask on the tensor's own device, otherwise indexing a GPU
        # tensor with a CPU mask fails.
        mask = (t.rand(edgeNum, device=vals.device) + keep_rate).floor().type(t.bool)
        newVals = vals[mask]
        if self.rescale:
            # Expected propagation then matches the full graph used at eval time.
            newVals = newVals / keep_rate
        newIdxs = idxs[:, mask]
        return t.sparse_coo_tensor(newIdxs, newVals, adj.shape)


def degree_sensitive_prune(trn_mat, rho, device=None, generator=None):
    """Phase 3: degree-sensitive edge pruning, resampled each epoch.

    Each interaction gets weight p_ui = 1/sqrt(w_u * w_i). Those weights are
    normalised into a distribution and (1 - rho) * |E| edges are sampled to be
    KEPT, without replacement, via Gumbel top-k. Low-degree edges carry high
    weight and are preferentially retained, so hub edges are preferentially
    dropped -- which is what the denoising argument calls for.

    Using p_ui directly as a drop probability instead would prune the sparse
    tail ~5x more than hubs and pin the rate near 5.7%, which is the opposite
    of the intent.

    The surviving edges are renormalised as D^-1/2 A D^-1/2 using the PRUNED
    degrees, matching the 1/sqrt(|N_rho(u)| |N_rho(i)|) message-passing weights.

    Args:
        trn_mat: scipy sparse interaction matrix (n_users x n_items).
        rho: fraction of edges to drop, in [0, 1).
        device: target device; defaults to utils.get_device().
        generator: torch.Generator for reproducible sampling.

    Returns:
        Coalesced sparse COO float32 bipartite adjacency of shape
        (n_users + n_items, n_users + n_items).
    """
    device = get_device() if device is None else device
    mat = trn_mat.tocoo()
    n_users, n_items = mat.shape
    n_nodes = n_users + n_items

    rows = t.from_numpy(mat.row.astype(np.int64))
    cols = t.from_numpy(mat.col.astype(np.int64))

    binary = (mat != 0)
    w_u = t.from_numpy(np.asarray(binary.sum(axis=1)).ravel().astype(np.float32))
    w_i = t.from_numpy(np.asarray(binary.sum(axis=0)).ravel().astype(np.float32))

    p = 1.0 / t.sqrt(w_u[rows].clamp(min=1.0) * w_i[cols].clamp(min=1.0))

    n_keep = round((1.0 - rho) * mat.nnz)
    if n_keep >= mat.nnz:
        keep = t.arange(mat.nnz)
    else:
        # Gumbel top-k == weighted sampling without replacement, and unlike
        # torch.multinomial(replacement=False) it is fast at this scale.
        u = t.rand(mat.nnz, generator=generator)
        keys = t.log(p) - t.log(-t.log(u))
        keep = keys.topk(n_keep).indices

    kr = rows[keep]
    kc = cols[keep] + n_users  # items occupy the upper block of the node index

    # Mirror each interaction so the bipartite adjacency is symmetric.
    idxs = t.stack([t.cat([kr, kc]), t.cat([kc, kr])]).to(device)
    vals = t.ones(idxs.shape[1], device=device)

    deg = t.zeros(n_nodes, device=device).index_add_(0, idxs[0], vals)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[t.isinf(d_inv_sqrt)] = 0.0
    vals = d_inv_sqrt[idxs[0]] * vals * d_inv_sqrt[idxs[1]]

    return t.sparse_coo_tensor(idxs, vals, (n_nodes, n_nodes)).coalesce()
