"""Compare SIG against the LightGCN baseline under an identical protocol.

Both models are trained to early stopping on validation recall@20, then the best
validation checkpoint is evaluated on the held-out test set. Same seed, same
dataset, same optimiser settings, same early-stopping rule.

Deliberately does NOT import main.py: that script writes final_*_embeddings.pkl
into the repo root as an import side effect.

Usage:
    python experiments/compare_sig_vs_lightgcn.py [--seed 42] [--dataset yelp]
                                                      [--models LightGCN SIG]
                                                      [--max-epochs 3000]
                                                      [--time-budget-min 60]
"""

import argparse
import contextlib
import functools
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config_manager import ConfigManager
from data_handler import DataHandler
from logger import Logger
from model_factory import ModelFactory
from trainer import Trainer
from utils import get_device


def init_seed(seed, device):
    """Replicates main.py's init_seed so runs match the project's real regime."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=device.type == "mps")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def _reproducibility_metadata(dataset, file_paths_items):
    """Fingerprint the exact code and inputs that produced a result.

    Git SHA alone is insufficient during thesis work because most experiments
    are launched from a dirty working tree.  The source digest includes the
    currently checked-out bytes, and the data digest includes interactions and
    semantic profiles.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_paths = [
        "config.yaml", "config_manager.py", "data_handler.py",
        "graph_builder.py", "loss_funcs.py", "metrics.py",
        "model_factory.py", "trainer.py", "utils.py",
        "experiments/compare_sig_vs_lightgcn.py",
    ]
    source_paths.extend(
        os.path.relpath(os.path.join(dp, fn), root)
        for dp, _, files in os.walk(os.path.join(root, "models"))
        for fn in files if fn.endswith(".py")
    )
    source_digest = hashlib.sha256()
    for rel in sorted(source_paths):
        path = os.path.join(root, rel)
        source_digest.update(rel.encode())
        source_digest.update(bytes.fromhex(_sha256(path)))

    input_paths = dict(file_paths_items)
    for name in ("usr_emb_np.pkl", "itm_emb_np.pkl"):
        path = os.path.join(root, "data", dataset, name)
        if os.path.exists(path):
            input_paths[name] = path
    data_hashes = {
        key: _sha256(os.path.abspath(path)) for key, path in input_paths.items()
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None

    return {
        "git_sha": git_sha,
        "source_sha256": source_digest.hexdigest(),
        "data_sha256": data_hashes,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def reproducibility_metadata(dataset, file_paths):
    return _reproducibility_metadata(dataset, tuple(sorted(file_paths.items())))


class BudgetedTrainer(Trainer):
    """Trainer with a wall-clock guard.

    A model stopped by the budget rather than by early stopping has not
    converged, and comparing it against one that did would be misleading. The
    guard records which happened so the report can say so plainly.
    """

    def __init__(self, *args, time_budget_s=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_budget_s = time_budget_s
        self.started = time.time()
        self.stopped_reason = "epoch_cap"
        self.epochs_run = 0

    def train_epoch(self, model, epoch_idx):
        self.epochs_run = epoch_idx + 1
        if self.time_budget_s and (time.time() - self.started) > self.time_budget_s:
            self.stopped_reason = "time_budget"
            raise TimeoutError("time budget exhausted")
        return super().train_epoch(model, epoch_idx)


def run_one(model_name, dataset, seed, max_epochs, time_budget_s, device, cm,
            overrides=None, batch_size=None, evaluate_test=True, patience=None):
    dc = dict(cm.get_dataset_config(dataset))
    dc["epoch"] = max_epochs
    if batch_size:
        dc["batch_size"] = batch_size
    if patience:
        dc["patience"] = patience
    fp = cm.get_file_paths(dc)

    init_seed(seed, device)

    dh = DataHandler(dc["user_num"], dc["item_num"], dataset,
                     1024, dc["batch_size"], device)
    dh.trn_file, dh.val_file, dh.tst_file = (
        fp["train_file"],
        fp["val_file"],
        fp["test_file"],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        dh.load_data()

    hp = dict(cm.get_model_hyperparameters(dc, model_name))
    for k, v in (overrides or {}).items():
        if k not in hp:
            raise KeyError(f"{model_name} has no hyperparameter {k!r}; has {sorted(hp)}")
        hp[k] = type(hp[k])(v)
    model = ModelFactory.create_model(model_name, dh, hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    logger = Logger(model_name, f"{dataset}_cmp_seed{seed}")
    tr = BudgetedTrainer(
        dh, dc["epoch"], dc["learning_rate"], dc["weight_decay"], dc["batch_size"],
        dc["patience"], logger, [5, 10, 20], device, model_name, dataset, model,
        time_budget_s=time_budget_s,
        save_checkpoint=False,
        test_at_end=False,
    )

    t0 = time.time()
    buf = io.StringIO()
    timed_out = False
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            tr.train(model)
    except TimeoutError:
        timed_out = True
    if evaluate_test:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            test_result = tr.test(model)
    else:
        test_result = None
    elapsed = time.time() - t0

    out = buf.getvalue()
    best_epoch = None
    val_recall20 = []
    for line in out.split("\n"):
        if line.startswith("Best Epoch"):
            best_epoch = int(line.split()[2].rstrip("."))
        if "Validation set" in line and "recall@20:" in line:
            val_recall20.append(float(line.split("recall@20:")[1].split()[0]))
    best_epoch = tr.best_epoch if tr.best_epoch is not None else best_epoch
    best_val = tr.best_metric_value
    if best_val is None:
        best_val = max(val_recall20) if val_recall20 else None

    train_batch_size = batch_size or dc["batch_size"]
    protocol = {
        "train_batch_size": train_batch_size,
        "test_batch_size": 1024,
        "max_epochs": max_epochs,
        "patience": dc["patience"],
        "eval_every": 3,
        "learning_rate": dc["learning_rate"],
        "weight_decay": dc["weight_decay"],
        "time_budget_s": time_budget_s,
        "device": str(device),
        "selection_metric": "validation_recall@20",
        "test_policy": (
            "once_after_fixed_config_at_best_validation_checkpoint"
            if evaluate_test else "not_evaluated_validation_only"
        ),
    }

    result = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "params": int(n_params),
        "hyperparameters": hp,
        "epochs_run": tr.epochs_run,
        "best_epoch": best_epoch,
        "stopped": "time_budget" if timed_out else tr.stopped_reason,
        "elapsed_s": round(elapsed, 1),
        "best_val_recall20": best_val,
        "protocol": protocol,
        "provenance": reproducibility_metadata(dataset, fp),
    }
    if test_result is not None:
        result["test"] = {
            k: [float(x) for x in v] for k, v in test_result.items()
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="yelp")
    ap.add_argument("--models", nargs="+", default=["LightGCN", "SIG"])
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--time-budget-min", type=float, default=60.0)
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--override", nargs="*", default=[],
                    help="hyperparameter overrides, e.g. --override layer_num=3 knn_k=20")
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the dataset's training batch size")
    args = ap.parse_args()

    device = get_device()
    cm = ConfigManager("config.yaml")
    os.makedirs(args.out, exist_ok=True)

    overrides = dict(o.split("=", 1) for o in args.override)
    if overrides:
        print(f"overrides: {overrides}", flush=True)
    print(f"device={device} dataset={args.dataset} seed={args.seed} "
          f"max_epochs={args.max_epochs} budget={args.time_budget_min}min/model", flush=True)

    results = []
    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        r = run_one(name, args.dataset, args.seed, args.max_epochs,
                    args.time_budget_min * 60, device, cm, overrides,
                    batch_size=args.batch_size)
        results.append(r)
        print(f"  epochs_run={r['epochs_run']} best_epoch={r['best_epoch']} "
              f"stopped={r['stopped']} elapsed={r['elapsed_s']}s params={r['params']:,}", flush=True)
        print(f"  best VAL recall@20       = {r['best_val_recall20']}", flush=True)
        print(f"  test recall@[5,10,20]    = {[round(x,4) for x in r['test']['recall']]}", flush=True)
        print(f"  test ndcg@[5,10,20]      = {[round(x,4) for x in r['test']['ndcg']]}", flush=True)
        print(f"  test precision@[5,10,20] = {[round(x,4) for x in r['test']['precision']]}", flush=True)

    # The filename carries the model names as well as the tag: naming by tag
    # alone lets two sweeps over different models collide silently, which is a
    # data-loss bug rather than an inconvenience.
    models_slug = "-".join(args.models)
    path = os.path.join(
        args.out, f"{args.dataset}_{models_slug}_seed{args.seed}{args.tag}.json")
    if os.path.exists(path):
        print(f"  NOTE: overwriting existing {os.path.basename(path)}", flush=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (test set, at best-validation checkpoint) ===", flush=True)
    hdr = f"{'model':14s} {'best_ep':>7s} {'recall@20':>10s} {'ndcg@20':>9s} {'stopped':>22s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        print(f"{r['model']:14s} {str(r['best_epoch']):>7s} "
              f"{r['test']['recall'][-1]:10.4f} {r['test']['ndcg'][-1]:9.4f} "
              f"{r['stopped']:>22s}", flush=True)
    if len(results) == 2:
        a, b = results
        d = b["test"]["recall"][-1] - a["test"]["recall"][-1]
        rel = 100 * d / a["test"]["recall"][-1]
        print(f"\n{b['model']} - {a['model']} recall@20: {d:+.4f} ({rel:+.2f}%)", flush=True)
        print("Single seed. Run-to-run noise on this hardware is ~1.2e-4 recall@20; "
              "a gap of that order is not a result.", flush=True)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
