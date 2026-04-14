from .cnn_baseline import CNNBaseline
from .vit import ViTRegressor
from .cosmo_mamba import CosmoMamba

MODEL_REGISTRY = {
    "cnn": CNNBaseline,
    "vit": ViTRegressor,
    "mamba": CosmoMamba,
}


def build_model(cfg):
    name = cfg["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**cfg["model"]["params"])
