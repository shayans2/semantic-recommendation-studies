import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pytest

from models.sig import SIG
from loss_funcs import cal_infonce_loss
from utils import spmm


class FakeDataHandler:
    """Minimal stand-in so the model is testable without the real dataset."""

    def __init__(self, n_users=40, n_items=30, sem_dim=16, seed=0):
        rng = np.random.RandomState(seed)
        self.device = torch.device("cpu")
        dense = (rng.rand(n_users, n_items) < 0.15).astype(np.float32)
        dense[np.arange(n_users), rng.randint(0, n_items, n_users)] = 1.0
        self.trn_mat = sp.coo_matrix(dense)
        self.user_num, self.item_num = self.trn_mat.shape
        self.item_sem_emb = rng.randn(n_items, sem_dim)
        from utils import degree_sensitive_prune
        self.torch_adj = degree_sensitive_prune(
            self.trn_mat, 0.0, torch.device("cpu"), torch.Generator().manual_seed(0)
        )


def _model(dh=None):
    dh = dh or FakeDataHandler()
    return SIG(dh, embedding_size=8, layer_num=2, ii_layer_num=1,
                      knn_k=5, rho=0.2, reg_weight=1e-7,
                      kd_weight=1e-2, kd_temperature=0.2)


def test_forward_shapes():
    dh = FakeDataHandler()
    m = _model(dh)
    u, i = m.forward(m.adj, 1.0)
    assert u.shape == (dh.user_num, 8)
    assert i.shape == (dh.item_num, 8)


def test_fusion_changes_item_embeddings():
    """h*_i = h_i + normalize(h~_i) must differ from the collaborative-only h_i."""
    m = _model()
    _, fused = m.forward(m.adj, 1.0)
    collab_only = m._collaborative_only_items()
    assert not torch.allclose(fused, collab_only, atol=1e-6)


def test_structural_component_is_unit_normalised():
    m = _model()
    h_struct = m._structural_items()
    norms = h_struct.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_cal_loss_returns_three_components():
    m = _model()
    batch = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 6])]
    loss, parts = m.cal_loss(batch)
    assert set(parts) == {"bpr_loss", "kd_loss", "reg_loss"}
    assert torch.isfinite(loss)


def test_cal_loss_is_differentiable():
    m = _model()
    batch = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 6])]
    loss, _ = m.cal_loss(batch)
    loss.backward()
    assert m.user_embeds.grad is not None
    assert m.item_embeds.grad is not None


def test_frozen_graph_receives_no_gradient():
    m = _model()
    assert not m.item_graph.requires_grad
    assert not any(p is m.item_graph for p in m.parameters())


def test_reset_cache_makes_load_state_dict_take_effect():
    """The bug this codebase already had once: a stale cache outliving the weights."""
    import copy
    m = _model()
    batch = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 6])]
    m.cal_loss(batch)
    mask = torch.zeros(2, m.item_num).long()
    before = m.full_predict([torch.tensor([0, 1]), mask]).clone()
    snapshot = copy.deepcopy(m.state_dict())
    with torch.no_grad():
        m.user_embeds *= 50.0
    m.reset_cache()
    changed = m.full_predict([torch.tensor([0, 1]), mask]).clone()
    assert not torch.allclose(before, changed)
    m.load_state_dict(snapshot)
    m.reset_cache()
    restored = m.full_predict([torch.tensor([0, 1]), mask])
    assert torch.allclose(before, restored, atol=1e-5)


def test_forward_does_not_cache_while_training():
    m = _model()
    m.is_training = True
    m.forward(m.adj, 0.8)
    assert m.final_embeds is None


def test_full_predict_masks_training_interactions():
    m = _model()
    mask = torch.zeros(2, m.item_num).long()
    mask[0, 3] = 1
    preds = m.full_predict([torch.tensor([0, 1]), mask])
    assert preds[0, 3] < -1e7


def test_on_epoch_start_resamples_graph():
    m = _model()
    first = m.train_adj.indices().clone()
    m.on_epoch_start()
    second = m.train_adj.indices()
    assert first.shape != second.shape or not torch.equal(first, second)


def test_evaluation_uses_full_graph_not_pruned():
    """Pruning is a training-time denoiser; evaluation must see every edge."""
    dh = FakeDataHandler()
    m = _model(dh)
    m.is_training = False
    assert m._select_adj() is m.adj


def test_structural_component_is_final_layer_not_mean():
    """Eq. 6 takes the FINAL layer, not a mean over layers. F.normalize alone
    forces unit norm regardless of which pooling was used, so pin the actual
    value against a hand-rolled expected computation instead."""
    dh = FakeDataHandler()
    m1 = SIG(dh, embedding_size=8, layer_num=2, ii_layer_num=1,
                     knn_k=5, rho=0.2, reg_weight=1e-7,
                     kd_weight=1e-2, kd_temperature=0.2)
    expected1 = F.normalize(spmm(m1.item_graph, m1.item_embeds), p=2, dim=1)
    assert torch.allclose(m1._structural_items(), expected1, atol=1e-6)

    m2 = SIG(dh, embedding_size=8, layer_num=2, ii_layer_num=2,
                     knn_k=5, rho=0.2, reg_weight=1e-7,
                     kd_weight=1e-2, kd_temperature=0.2)
    h = spmm(m2.item_graph, m2.item_embeds)
    h = spmm(m2.item_graph, h)
    expected2 = F.normalize(h, p=2, dim=1)
    assert torch.allclose(m2._structural_items(), expected2, atol=1e-6)


def test_collaborative_mean_pool_averages_all_layers_including_zero():
    """Eqs. 9-10 average layers 0..L_ui inclusive, i.e. divisor layer_num + 1.
    A divisor of layer_num, or dropping layer 0 from the sum, would keep the
    rest of the suite green, so pin the exact hand-rolled expectation."""
    m = _model()
    adj = m._select_adj()
    embeds = torch.concat([m.user_embeds, m.item_embeds], dim=0)
    embeds_list = [embeds]
    for _ in range(m.layer_num):
        embeds_list.append(spmm(adj, embeds_list[-1]))
    assert len(embeds_list) == m.layer_num + 1
    expected = sum(embeds_list) / len(embeds_list)

    actual = m._collaborative(adj)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_forward_dispatches_pruned_vs_full_graph_by_training_flag():
    """Exercise the real call path (forward() itself, not _select_adj()
    directly): training must propagate over the pruned graph and evaluation
    over the full graph. A forward() that hardcoded self.adj regardless of
    is_training would pass every other test in this suite while silently
    disabling Phase 3 pruning."""
    dh = FakeDataHandler()
    m = _model(dh)  # rho=0.2, so train_adj is a genuine subset of adj
    assert m.train_adj._indices().shape[1] != m.adj._indices().shape[1]

    m.is_training = True
    u_train, i_train = m.forward()

    # Clear any cache state before the eval call so the only thing that can
    # differ between the two calls is which adjacency was selected -- not a
    # leftover cached value from the training call (forward() never caches
    # while is_training is True, but reset_cache() here removes any doubt).
    m.reset_cache()
    m.is_training = False
    u_eval, i_eval = m.forward()

    assert not torch.allclose(u_train, u_eval, atol=1e-6)
    assert not torch.allclose(i_train, i_eval, atol=1e-6)


def test_kd_loss_anchors_fused_positive_items_only():
    """Eq. 14: InfoNCE over POSITIVE items only, computed on the FUSED item
    embedding, with in-batch negatives. Anchoring users too, or anchoring the
    pre-fusion collaborative vector instead of the fused one, is exactly the
    Phase-5 defect this model exists to fix -- pin the exact value so either
    mistake fails the test."""
    m = _model()
    batch = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 6])]
    loss, parts = m.cal_loss(batch)

    # cal_loss leaves is_training=True and never re-caches, so recomputing
    # forward here (no backward/step happened in between) reproduces the
    # exact fused embedding cal_loss used internally.
    with torch.no_grad():
        _, fused_items = m.forward(m._select_adj())
    pos_embeds = fused_items[batch[1]]
    posprf_embeds = m.mlp(m.item_sem_emb[batch[1]])
    expected_kd = cal_infonce_loss(
        pos_embeds, posprf_embeds, posprf_embeds, m.kd_temperature
    )
    expected_kd = (expected_kd / pos_embeds.shape[0]) * m.kd_weight

    assert torch.allclose(parts["kd_loss"], expected_kd, atol=1e-5)


def test_pos_neg_anchor_adds_exact_negative_item_term():
    from models.sig_ablations import SIG_PosNegAnchor

    dh = FakeDataHandler()
    kwargs = dict(
        embedding_size=8, layer_num=2, ii_layer_num=1, knn_k=5, rho=0.2,
        reg_weight=1e-7, kd_weight=1e-2, kd_temperature=0.2,
    )
    m = SIG_PosNegAnchor(dh, **kwargs)
    batch = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 6])]
    _, parts = m.cal_loss(batch)

    with torch.no_grad():
        _, fused_items = m.forward(m._select_adj())
    pos, neg = fused_items[batch[1]], fused_items[batch[2]]
    pos_sem = m.mlp(m.item_sem_emb[batch[1]])
    neg_sem = m.mlp(m.item_sem_emb[batch[2]])
    expected = (
        cal_infonce_loss(pos, pos_sem, pos_sem, m.kd_temperature)
        + cal_infonce_loss(neg, neg_sem, neg_sem, m.kd_temperature)
    ) / pos.shape[0] * m.kd_weight

    assert torch.allclose(parts["kd_loss"], expected, atol=1e-5)


def test_permuted_graph_does_not_change_global_rng_after_model_init():
    """The graph control must not alter later DataLoader shuffling.

    Base and permuted models consume the same global RNG during ordinary model
    initialisation. The permutation itself must use a private generator.
    """
    from models.sig_ablations import SIG_PermutedGraph

    kwargs = dict(
        embedding_size=8, layer_num=2, ii_layer_num=1, knn_k=5, rho=0.2,
        reg_weight=1e-7, kd_weight=1e-2, kd_temperature=0.2,
    )
    torch.manual_seed(1234)
    SIG(FakeDataHandler(), **kwargs)
    state_after_base = torch.get_rng_state()

    torch.manual_seed(1234)
    SIG_PermutedGraph(FakeDataHandler(), **kwargs)
    state_after_control = torch.get_rng_state()

    assert torch.equal(state_after_base, state_after_control)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_cpu_mps_parity():
    import copy
    dh = FakeDataHandler()
    cpu_model = _model(dh)
    snapshot = copy.deepcopy(cpu_model.state_dict())
    cpu_model.reset_cache()
    with torch.no_grad():
        u_cpu, i_cpu = cpu_model.forward(cpu_model.adj, 1.0)

    dh_mps = FakeDataHandler()
    dh_mps.device = torch.device("mps")
    dh_mps.torch_adj = dh_mps.torch_adj.to("mps")
    mps_model = _model(dh_mps).to("mps")
    mps_model.load_state_dict({k: v.to("mps") for k, v in snapshot.items()})
    mps_model.reset_cache()
    with torch.no_grad():
        u_mps, i_mps = mps_model.forward(mps_model.adj, 1.0)

    assert torch.allclose(u_cpu, u_mps.cpu(), atol=1e-5)
    assert torch.allclose(i_cpu, i_mps.cpu(), atol=1e-5)
