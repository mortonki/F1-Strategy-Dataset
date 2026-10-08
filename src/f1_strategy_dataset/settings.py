"""Centralized settings for the F1 Strategy Prediction pipeline.

Loads configuration from ``config.yaml`` and initializes MLflow experiment
tracking so that the tracking URI is consistent across the whole project.
"""

import os
import yaml
import mlflow

# ``config.yaml`` lives at the project root, two levels up from this module.
_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.yaml")
)

# Fallback tracking URI used when the config does not specify one.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# Scheme prefix for local sqlite tracking stores.
_SQLITE_SCHEME = "sqlite:///"


def load_config(config_path: str | None = None) -> dict:
    """Load the YAML configuration file.

    Args:
        config_path: Path to the config file. Defaults to the repo-root
            ``config.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    path = config_path or _CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_tracking_uri(config: dict) -> str:
    """Return the raw MLflow tracking URI from the config.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        The tracking URI string from config, or the default if absent.
    """
    return config.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def init_mlflow(config: dict) -> str:
    """Set the MLflow tracking URI and experiment from the config.

    Relative tracking URIs are resolved against the project root so the
    pipeline behaves the same regardless of the current working directory.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        The resolved tracking URI (with the ``sqlite:///`` scheme).
    """
    raw_uri = get_tracking_uri(config)

    # Strip a leading sqlite:// scheme if present, then resolve relative paths.
    db_path = raw_uri[len(_SQLITE_SCHEME):] if raw_uri.startswith(_SQLITE_SCHEME) else raw_uri
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(_CONFIG_PATH), db_path)

    tracking_uri = f"{_SQLITE_SCHEME}{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("F1 Strategy Prediction")
    return tracking_uri