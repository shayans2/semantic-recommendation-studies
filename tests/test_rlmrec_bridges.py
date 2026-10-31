import torch

from loss_funcs import cal_infonce_loss
from models.sig_ablations import SIG_NoStructural
from models.rlmrec_bridges import (
    RLMRecItemOnlyCombinedBridge,
    RLMRecItemOnlyDegreePrune,
    RLMRecItemOnlyMeanAggregation,
    RLMRecItemOnlyPositiveOnly,
)
from models.rlmrec_faithful import LightGCN_plus_RLMRec_ItemOnly
from tests.test_rlmrec_item_only import FakeDataHandler


def kwargs(kd_weight=0.003):
    return {
        "embedding_size": 4,
        "keep_rate": 1.0,
        "layer_num": 2,
        "reg_weight": 0.0,
        "kd_weight": kd_weight,
        "kd_temperature": 0.2,
    }


def batch():
    return [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6]),
    ]


def test_positive_only_bridge_removes_exactly_negative_item_term():
    model = RLMRecItemOnlyPositiveOnly(FakeDataHandler(), **kwargs())
    _, parts = model.cal_loss(batch())

    with torch.no_grad():
        users, items = model.forward(model.adj, model.keep_rate)
        _, positives, _ = model._pick(users, items, batch())
        projected = model.mlp(model.itmprf_embeds)
        _, projected_positives, _ = model._pick(projected, projected, batch())
        expected = cal_infonce_loss(
            positives,
            projected_positives,
            projected_positives,
            model.kd_temperature,
        ) / len(batch()[0]) * model.kd_weight

    assert torch.allclose(parts["kd_loss"], expected, atol=1e-6)


def test_mean_bridge_differs_from_baseline_only_by_layer_count_scale_at_eval():
    handler = FakeDataHandler()
    baseline = LightGCN_plus_RLMRec_ItemOnly(handler, **kwargs())
    bridge = RLMRecItemOnlyMeanAggregation(handler, **kwargs())
    bridge.load_state_dict(baseline.state_dict())
    baseline.eval()
    bridge.eval()

    base_users, base_items = baseline.forward()
    mean_users, mean_items = bridge.forward()
    scale = baseline.layer_num + 1

    assert torch.allclose(mean_users, base_users / scale, atol=1e-6)
    assert torch.allclose(mean_items, base_items / scale, atol=1e-6)


def test_degree_prune_bridge_does_not_advance_global_rng_during_setup():
    handler = FakeDataHandler()
    torch.manual_seed(1234)
    LightGCN_plus_RLMRec_ItemOnly(handler, **kwargs())
    baseline_state = torch.get_rng_state()

    torch.manual_seed(1234)
    bridge = RLMRecItemOnlyDegreePrune(handler, rho=0.2, **kwargs())
    bridge_state = torch.get_rng_state()

    assert torch.equal(bridge_state, baseline_state)
    assert bridge.train_adj._nnz() == 2 * round(0.8 * handler.trn_mat.nnz)


def test_combined_bridge_matches_sig_without_structural_branch():
    handler = FakeDataHandler()
    bridge = RLMRecItemOnlyCombinedBridge(
        handler, rho=0.2, **kwargs(kd_weight=0.03)
    )
    sig = SIG_NoStructural(
        handler,
        embedding_size=4,
        layer_num=2,
        ii_layer_num=1,
        knn_k=3,
        rho=0.2,
        reg_weight=0.0,
        kd_weight=0.03,
        kd_temperature=0.2,
    )
    sig.load_state_dict(bridge.state_dict())
    sig.train_adj = bridge.train_adj

    bridge_loss, bridge_parts = bridge.cal_loss(batch())
    sig_loss, sig_parts = sig.cal_loss(batch())

    assert torch.allclose(bridge_loss, sig_loss, atol=1e-6)
    assert bridge_parts.keys() == sig_parts.keys()
    for key in bridge_parts:
        assert torch.allclose(bridge_parts[key], sig_parts[key], atol=1e-6)


def test_factory_exposes_all_amazon_bridge_configs():
    from config_manager import ConfigManager
    from model_factory import ModelFactory

    names = {
        "RLMRec_Bridge_PosOnly_64",
        "RLMRec_Bridge_Mean_64",
        "RLMRec_Bridge_DegreePrune_64",
        "RLMRec_Bridge_Combined_64",
    }
    assert names.issubset(ModelFactory.available_models())
    cm = ConfigManager("config.yaml")
    dc = cm.get_dataset_config("amazon")
    for name in names:
        assert cm.get_model_hyperparameters(dc, name)["embedding_size"] == 64
