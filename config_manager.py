import yaml
import os

from model_factory import ModelFactory


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_dataset_config(self, dataset_name):
        return self.config["datasets"][dataset_name]

    def get_model_hyperparameters(self, dataset_config, model_name):
        model_name = ModelFactory.canonical_name(model_name)
        if model_name not in dataset_config["models"]:
            raise ValueError(
                f"Model '{model_name}' not configured for dataset '{dataset_config}'"
            )
        return dataset_config["models"][model_name]["hyperparameters"]

    def get_file_paths(self, dataset_config):
        data_dir = dataset_config["data_dir"]
        return {
            "train_file": os.path.join(data_dir, dataset_config["train_file"]),
            "val_file": os.path.join(data_dir, dataset_config["val_file"]),
            "test_file": os.path.join(data_dir, dataset_config["test_file"]),
        }
