import numpy as np
import scipy.sparse as sp
import torch

from loss_funcs import cal_infonce_loss
from models.rlmrec_faithful import LightGCN_plus_RLMRec_ItemOnly


class FakeDataHandler:
    def __init__(self, with_user_profiles=True):
        rng = np.random.RandomState(7)
        self.user_num = 5
        self.item_num = 7
        self.device = torch.device("cpu")
        self.item_sem_emb = rng.randn(self.item_num, 11).astype(np.float32)
        self.user_sem_emb = (
            rng.randn(self.user_num, 11).astype(np.float32)
            if with_user_profiles else None
        )
        self.trn_mat = sp.coo_matrix(
            (
                np.ones(10, dtype=np.float32),
                (
                    np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
                    np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5]),
                ),
            ),
            shape=(self.user_num, self.item_num),
        ).tocsr()
        eye = sp.eye(self.user_num + self.item_num, format="coo", dtype=np.float32)
        self.torch_adj = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([eye.row, eye.col]), dtype=torch.long),
            torch.tensor(eye.data),
            eye.shape,
        ).coalesce()


def make_model(data_handler):
    return LightGCN_plus_RLMRec_ItemOnly(
        data_handler,
        embedding_size=4,
        keep_rate=1.0,
        layer_num=1,
        reg_weight=0.0,
        kd_weight=0.01,
        kd_temperature=0.2,
    )


def test_item_only_model_does_not_require_user_profiles():
    model = make_model(FakeDataHandler(with_user_profiles=False))
    assert not hasattr(model, "usrprf_embeds")


def test_item_only_kd_is_exactly_positive_plus_negative_item_terms():
    model = make_model(FakeDataHandler())
    batch = [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6]),
    ]

    _, parts = model.cal_loss(batch)
    with torch.no_grad():
        users, items = model.forward(model.adj, model.keep_rate)
        _, positives, negatives = model._pick(users, items, batch)
        projected_items = model.mlp(model.itmprf_embeds)
        _, projected_positives, projected_negatives = model._pick(
            projected_items, projected_items, batch
        )
        expected = (
            cal_infonce_loss(
                positives,
                projected_positives,
                projected_positives,
                model.kd_temperature,
            )
            + cal_infonce_loss(
                negatives,
                projected_negatives,
                projected_negatives,
                model.kd_temperature,
            )
        ) / len(batch[0]) * model.kd_weight

    assert torch.allclose(parts["kd_loss"], expected, atol=1e-6)


def test_factory_configs_item_only_at_both_dimensions_and_datasets():
    from config_manager import ConfigManager
    from model_factory import ModelFactory

    cm = ConfigManager("config.yaml")
    assert {"RLMRec_ItemOnly_32", "RLMRec_ItemOnly_64"}.issubset(
        ModelFactory.available_models()
    )
    for dataset in ("amazon", "yelp"):
        dc = cm.get_dataset_config(dataset)
        hp32 = cm.get_model_hyperparameters(dc, "RLMRec_ItemOnly_32")
        hp64 = cm.get_model_hyperparameters(dc, "RLMRec_ItemOnly_64")
        assert hp32["embedding_size"] == 32
        assert hp64["embedding_size"] == 64
