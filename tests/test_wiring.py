import torch

from model_factory import ModelFactory
from config_manager import ConfigManager


def test_factory_registers_sig():
    assert "SIG" in ModelFactory.available_models()


def test_config_has_sig_for_both_datasets():
    cm = ConfigManager("config.yaml")
    for ds in ("yelp", "amazon"):
        hp = cm.get_model_hyperparameters(cm.get_dataset_config(ds), "SIG")
        for key in ("embedding_size", "layer_num", "ii_layer_num", "knn_k",
                    "rho", "reg_weight", "kd_weight", "kd_temperature"):
            assert key in hp, f"{ds} missing {key}"


def test_hook_is_inert_for_baselines():
    """on_epoch_start must not exist on any baseline model class.

    Excludes the SIG family by type rather than by name: its ablation
    subclasses inherit the hook legitimately, and a name check silently missed
    them.
    """
    import importlib
    from model_factory import _MODEL_DEFINITIONS
    from models.sig import SIG

    checked = 0
    for name, info in _MODEL_DEFINITIONS.items():
        cls = getattr(importlib.import_module(info["module"]), info["class"])
        if issubclass(cls, SIG) or getattr(
            cls, "resamples_graph_each_epoch", False
        ):
            continue
        checked += 1
        assert not hasattr(cls, "on_epoch_start"), (
            f"{name} unexpectedly defines on_epoch_start"
        )
    assert checked >= 6, f"expected to check at least 6 baselines, checked {checked}"


class _FakeDataset:
    def sample_negs(self):
        pass


class _FakeDataLoader:
    """Yields a KNOWN number of batches (> 1) so per-epoch and per-batch
    invocation counts are distinguishable."""

    def __init__(self, num_batches):
        self._batches = [
            (torch.zeros(2, dtype=torch.long),) for _ in range(num_batches)
        ]
        self.dataset = _FakeDataset()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _FakeDataHandler:
    def __init__(self, num_batches):
        self.train_dataloader = _FakeDataLoader(num_batches)


class _FakeLogger:
    def log_loss(self, epoch_idx, loss_log_dict):
        pass


class _FakeModel(torch.nn.Module):
    """Minimal stand-in exposing exactly the surface Trainer.train_epoch
    touches: train(), cal_loss(batch), reset_cache(), parameters() (via
    nn.Module), plus on_epoch_start() with an invocation counter."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.epoch_start_calls = 0

    def cal_loss(self, batch_data):
        loss = self.weight.pow(2).sum()
        return loss, {"loss": loss}

    def on_epoch_start(self):
        self.epoch_start_calls += 1

    def reset_cache(self):
        pass


def test_hook_fires_once_per_epoch_not_per_batch():
    """A hook mis-wired inside the per-batch loop would still contain the
    substrings the source-grep test checks for, but would fire once per
    batch instead of once per epoch. This test catches that mutation
    behaviourally."""
    from trainer import Trainer

    num_batches = 3
    model = _FakeModel()
    trainer_obj = Trainer(
        data_handler=_FakeDataHandler(num_batches),
        epoch=1,
        lr=0.01,
        weight_decay=0,
        batch_size=2,
        patience=5,
        logger=_FakeLogger(),
        k=[5, 10, 20],
        device="cpu",
        model_name="FAKE",
        dataset_name="fake",
        model=model,
    )
    trainer_obj.create_optimizer(model)

    trainer_obj.train_epoch(model, 0)

    assert model.epoch_start_calls == 1
