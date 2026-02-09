"""
YAML/JSON configuration system for Atomic-1Bit models.

Usage:
    from atomic_1bit.config import load_config, config_to_atomic
    cfg = load_config("configs/flagship_12m.yaml")
    model_config = config_to_atomic(cfg)
"""
import json
import os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from atomic_1bit.model.transformer import AtomicConfig


def load_config(path: str) -> dict:
    """Load a YAML or JSON config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML configs: pip install pyyaml")
            return yaml.safe_load(f)
        elif path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")


def config_to_atomic(cfg: dict) -> AtomicConfig:
    """Convert a config dict to an AtomicConfig."""
    model = cfg.get("model", {})
    return AtomicConfig(
        vocab_size=model.get("vocab_size", 50257),
        dim=model.get("dim", 512),
        depth=model.get("depth", 8),
        heads=model.get("heads", 8),
        context_length=model.get("context_length", 1024),
    )


def get_training_config(cfg: dict) -> dict:
    """Extract training configuration."""
    return cfg.get("training", {})


def get_quantization_config(cfg: dict) -> dict:
    """Extract quantization configuration."""
    return cfg.get("quantization", {})
