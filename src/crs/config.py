"""Loading of the YAML configs under ``configs/``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("CRS_CONFIG_DIR", ROOT / "configs"))
DATA_DIR = Path(os.environ.get("CRS_DATA_DIR", ROOT / "data"))
WORDLIST_DIR = Path(__file__).resolve().parent / "wordlists"


def load_yaml(name: str | Path) -> dict[str, Any]:
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    key: str
    hf_id: str | None
    family: str
    dtype: str
    think_end: str
    chat_template_kwargs: dict[str, Any]
    sampling: dict[str, Any]
    tokenizer_id: str | None = None
    hf_batch_size: int = 16

    @property
    def tokenizer_path(self) -> str:
        return self.tokenizer_id or self.hf_id  # type: ignore[return-value]


@dataclass
class Setting:
    name: str
    family: str
    severity: float
    backend: str = "vllm"
    weight: dict[str, Any] | None = None
    kv: dict[str, Any] | None = None

    @property
    def changes_weights(self) -> bool:
        return self.weight is not None


@dataclass
class Fix:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


def models() -> dict[str, ModelConfig]:
    cfg = load_yaml("models.yaml")
    return {
        k: ModelConfig(key=k, tokenizer_id=v.get("tokenizer_id"), hf_batch_size=v.get("hf_batch_size", 16), **{
            kk: v[kk] for kk in ("hf_id", "family", "dtype", "think_end", "chat_template_kwargs", "sampling")
        })
        for k, v in cfg["models"].items()
    }


def generation_defaults() -> dict[str, Any]:
    cfg = load_yaml("models.yaml")
    return {"prompt_suffix": cfg["prompt_suffix"], **cfg["generation"]}


def settings() -> dict[str, Setting]:
    cfg = load_yaml("settings.yaml")
    return {s["name"]: Setting(**s) for s in cfg["settings"]}


def calibration() -> dict[str, Any]:
    return load_yaml("settings.yaml").get("calibration", {})


def fixes() -> dict[str, Fix]:
    cfg = load_yaml("fixes.yaml")
    out = {}
    for name, spec in cfg["fixes"].items():
        spec = dict(spec)
        out[name] = Fix(name=name, type=spec.pop("type"), params=spec)
    return out


def budgets() -> list[int]:
    return list(load_yaml("fixes.yaml").get("budgets", []))
