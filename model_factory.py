import importlib

# (module path, class name) for each registered model. Imported lazily so that a
# variant with heavy or optional dependencies (e.g. torchvision) cannot break
# the import of every other model.
_MODEL_DEFINITIONS = {
    "LightGCN": {
        "module": "models.lightgcn",
        "class": "LightGCN",
        "params": ["embedding_size", "keep_rate", "layer_num", "reg_weight"],
    },
    "LightGCN_plus": {
        "module": "models.lightgcn_plus",
        "class": "LightGCN_plus",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "LightGCN_latest": {
        "module": "models.lightgcn_latest",
        "class": "LightGCN_plus",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "LightGCN_multimodal": {
        "module": "models.lightgcn_multimodal",
        "class": "LightGCN",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "LightGCN_image": {
        "module": "models.lightgcn_image",
        "class": "LightGCN_plus",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "LightGCN_lattice": {
        "module": "models.lightgcn_lattice",
        "class": "LightGCN_plus_LATTICE",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "SIG": {
        "module": "models.sig",
        "class": "SIG",
        "params": [
            "embedding_size",
            "layer_num",
            "ii_layer_num",
            "knn_k",
            "rho",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "LightGCN_plus_RLMRec": {
        "module": "models.lightgcn_plus_rlmrec",
        "class": "LightGCN_plus_RLMRec",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "SIG_RandomGraph": {
        "module": "models.sig_ablations",
        "class": "SIG_RandomGraph",
        "params": [
            "embedding_size",
            "layer_num",
            "ii_layer_num",
            "knn_k",
            "rho",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "SIG_NoStructural": {
        "module": "models.sig_ablations",
        "class": "SIG_NoStructural",
        "params": [
            "embedding_size",
            "layer_num",
            "ii_layer_num",
            "knn_k",
            "rho",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "SIG_NoPrune": {
        "module": "models.sig_ablations",
        "class": "SIG_NoPrune",
        "params": [
            "embedding_size",
            "layer_num",
            "ii_layer_num",
            "knn_k",
            "rho",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "SIG_PermutedGraph": {
        "module": "models.sig_ablations",
        "class": "SIG_PermutedGraph",
        "params": [
            "embedding_size",
            "layer_num",
            "ii_layer_num",
            "knn_k",
            "rho",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "SIG_PosNegAnchor": {
        "module": "models.sig_ablations",
        "class": "SIG_PosNegAnchor",
        "params": [
            "embedding_size",
            "layer_num",
            "ii_layer_num",
            "knn_k",
            "rho",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "RLMRec_Coverage": {
        "module": "models.sig_ablations",
        "class": "LightGCN_plus_RLMRec_Coverage",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
            "profile_coverage",
        ],
    },
    "LightGCN_RLMRec_Exact": {
        "module": "models.rlmrec_faithful",
        "class": "LightGCN_RLMRec",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
        ],
    },
    "LightGCN_plus_RLMRec_Exact": {
        "module": "models.rlmrec_faithful",
        "class": "LightGCN_plus_RLMRec_Exact",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "LightGCN_RLMRec_64": {
        "module": "models.rlmrec_faithful",
        "class": "LightGCN_RLMRec",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
        ],
    },
    "LightGCN_plus_RLMRec_64": {
        "module": "models.rlmrec_faithful",
        "class": "LightGCN_plus_RLMRec_Exact",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "RLMRec_ItemOnly_32": {
        "module": "models.rlmrec_faithful",
        "class": "LightGCN_plus_RLMRec_ItemOnly",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "RLMRec_ItemOnly_64": {
        "module": "models.rlmrec_faithful",
        "class": "LightGCN_plus_RLMRec_ItemOnly",
        "params": [
            "embedding_size",
            "keep_rate",
            "layer_num",
            "reg_weight",
            "kd_weight",
            "kd_temperature",
        ],
    },
    "RLMRec_Bridge_PosOnly_64": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyPositiveOnly",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature",
        ],
    },
    "RLMRec_Bridge_Mean_64": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyMeanAggregation",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature",
        ],
    },
    "RLMRec_Bridge_DegreePrune_64": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyDegreePrune",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature", "rho",
        ],
    },
    "RLMRec_Bridge_Combined_64": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyCombinedBridge",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature", "rho",
        ],
    },
    # 32D twins of the bridges above for the published-protocol yelp
    # decomposition. Same classes; the suffix only selects the config entry.
    "RLMRec_Bridge_PosOnly_32": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyPositiveOnly",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature",
        ],
    },
    "RLMRec_Bridge_Mean_32": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyMeanAggregation",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature",
        ],
    },
    "RLMRec_Bridge_DegreePrune_32": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyDegreePrune",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature", "rho",
        ],
    },
    "RLMRec_Bridge_Combined_32": {
        "module": "models.rlmrec_bridges",
        "class": "RLMRecItemOnlyCombinedBridge",
        "params": [
            "embedding_size", "keep_rate", "layer_num", "reg_weight",
            "kd_weight", "kd_temperature", "rho",
        ],
    },
}


# Historical name of the SIG models, kept so that results, manifests and
# configs recorded before the rename still resolve to the current classes.
_LEGACY_MODEL_NAMES = {
    name.replace("SIG", "FREEDOM_RLM", 1): name
    for name in _MODEL_DEFINITIONS
    if name.startswith("SIG")
}


class ModelFactory:
    @staticmethod
    def available_models():
        return list(_MODEL_DEFINITIONS.keys())

    @staticmethod
    def canonical_name(model_name):
        """Map a historical model name onto its current name."""
        return _LEGACY_MODEL_NAMES.get(model_name, model_name)

    @staticmethod
    def create_model(model_name, data_handler, hyperparameters):
        model_name = ModelFactory.canonical_name(model_name)
        if model_name not in _MODEL_DEFINITIONS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: "
                f"{ModelFactory.available_models()}"
            )

        model_info = _MODEL_DEFINITIONS[model_name]
        model_class = getattr(
            importlib.import_module(model_info["module"]), model_info["class"]
        )

        missing = [p for p in model_info["params"] if p not in hyperparameters]
        if missing:
            raise ValueError(
                f"Model '{model_name}' requires hyperparameters {missing}, which "
                f"are not present in config.yaml for this dataset."
            )

        model_params = {p: hyperparameters[p] for p in model_info["params"]}
        return model_class(data_handler=data_handler, **model_params)
