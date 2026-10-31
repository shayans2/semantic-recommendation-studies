import os

# Must be set before the CUDA context is created, hence before anything touches torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pickle
import random

import numpy as np
import torch

from trainer import Trainer
from data_handler import DataHandler
from config_manager import ConfigManager
from model_factory import ModelFactory
from logger import Logger
from utils import get_device

config_manager = ConfigManager(config_path="config.yaml")
dataset_name = "yelp"
model_name = "LightGCN"
dataset_config = config_manager.get_dataset_config(dataset_name)
file_paths = config_manager.get_file_paths(dataset_config)
hyperparameters = config_manager.get_model_hyperparameters(dataset_config, model_name)


def init_seed(seed=42, device=None):
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

    # Metal does not provide deterministic kernels for every op, so ask for
    # determinism but warn instead of aborting the run there.
    warn_only = device is not None and device.type == "mps"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


device = get_device()
init_seed(device=device)

print(f"Using device: {device}")

logger = Logger(model_name=model_name, dataset_name=dataset_name)
logger.log(f"Device: {device} | model: {model_name} | dataset: {dataset_name}")

data_handler = DataHandler(
    user_num=dataset_config["user_num"],
    item_num=dataset_config["item_num"],
    dataset_name=dataset_name,
    test_batch_size=1024,
    train_batch_size=1024,
    device=device,
)

data_handler.trn_file = file_paths["train_file"]
data_handler.val_file = file_paths["val_file"]
data_handler.tst_file = file_paths["test_file"]
data_handler.load_data()

model = ModelFactory.create_model(model_name, data_handler, hyperparameters).to(device)

trainer = Trainer(
    data_handler=data_handler,
    epoch=dataset_config["epoch"],
    lr=dataset_config["learning_rate"],
    weight_decay=dataset_config["weight_decay"],
    batch_size=dataset_config["batch_size"],
    patience=dataset_config["patience"],
    logger=logger,
    k=[5, 10, 20],
    device=device,
    model_name=model_name,
    dataset_name=dataset_name,
    model=model,
)

trainer.train(model)

# `model` now holds the best checkpoint (Trainer restores it). Recompute the
# embeddings explicitly rather than reading a cached attribute, so what gets
# written to disk is guaranteed to match those weights.
model.eval()
model.reset_cache()
with torch.no_grad():
    final_user_embeds, final_item_embeds = model.forward(model.adj, 1.0)[:2]

user_embeds_np = final_user_embeds.cpu().detach().numpy()
item_embeds_np = final_item_embeds.cpu().detach().numpy()

# Save embeddings as .pkl files
with open("final_user_embeddings.pkl", "wb") as f:
    pickle.dump(user_embeds_np, f)

with open("final_item_embeddings.pkl", "wb") as f:
    pickle.dump(item_embeds_np, f)

print("User and item embeddings saved as .pkl files.")
