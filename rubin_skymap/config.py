"""Configuration loader for rubin-skymap.

Loads ``configs/base.yaml`` then deep-merges ``configs/dev.yaml`` (or the
file named by ``RUBIN_SKYMAP_ENV``) on top.  Also loads ``.env`` via
python-dotenv so that environment variables are available before any other
module imports.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_CONFIGS_DIR: Path = _REPO_ROOT / "configs"

# Load .env early; do not fail if file is absent.
load_dotenv(_REPO_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Parameters
    ----------
    base:
        The baseline dictionary.
    override:
        Values that take precedence over *base*.

    Returns
    -------
    dict
        A new dict with *override* values layered on top of *base*.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict.

    Parameters
    ----------
    path:
        Filesystem path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents, or an empty dict if the file is absent.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load and merge configuration files.

    Reads ``configs/base.yaml`` unconditionally, then deep-merges the
    environment-specific overlay chosen by the ``RUBIN_SKYMAP_ENV``
    environment variable (default: ``"dev"``).

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    base = _load_yaml(_CONFIGS_DIR / "base.yaml")
    env_name = os.environ.get("RUBIN_SKYMAP_ENV", "dev")
    overlay_path = _CONFIGS_DIR / f"{env_name}.yaml"
    overlay = _load_yaml(overlay_path)
    return _deep_merge(base, overlay)


def cfg_get(cfg: dict, dotted_path: str, default: Any = None) -> Any:
    """Retrieve a value from a nested dict using a dot-separated key path.

    Parameters
    ----------
    cfg:
        The configuration dictionary to traverse.
    dotted_path:
        A dot-separated string like ``"ingest.fink.timeout_sec"``.
    default:
        Value returned when the key path is not found.

    Returns
    -------
    Any
        The value at the requested path, or *default* if absent.

    Examples
    --------
    >>> cfg_get({"a": {"b": 1}}, "a.b")
    1
    >>> cfg_get({"a": {}}, "a.b.c", default=42)
    42
    """
    parts = dotted_path.split(".")
    node: Any = cfg
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part, default)
        if node is default:
            return default
    return node
