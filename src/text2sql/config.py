"""Central settings: paths, seed, model IDs, decoding params, run logging.

Every run should call `save_run_config(...)` so report tables can be traced
back to the exact settings that produced them.
"""

from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SPIDER_DIR = DATA_DIR / "spider"
EVAL_SUBSET_IDS = DATA_DIR / "eval_subset_ids.json"
OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EVAL_SCRIPT_DIR = REPO_ROOT / "test-suite-sql-eval"

# ---------------------------------------------------------------- seed
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- models
@dataclass(frozen=True)
class ModelSpec:
    key: str  # short name used for output dirs
    hf_id: str
    category: str  # "A" (no reasoning) or "B" (reasoning)
    params: str  # as reported in the methodology
    reasoning: bool = False  # True -> prepend <|think|> to system prompt
    gated: bool = False


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("gemma3-1b", "google/gemma-3-1b-it", "A", "~1.0B", gated=True),
        ModelSpec("gemma3-4b", "google/gemma-3-4b-it", "A", "~4.3B", gated=True),
        ModelSpec(
            "gemma4-e4b",
            "google/gemma-4-E4B-it",
            "B",
            "~8.0B total / ~4.5B effective (PLE)",
            reasoning=True,
        ),
    ]
}

# Same precision for all models where possible (E4B in bf16 won't fit a T4).
QUANTIZATION = "4bit-nf4"  # one of: "bf16", "8bit", "4bit-nf4"


# ---------------------------------------------------------------- decoding
@dataclass
class DecodingConfig:
    do_sample: bool = False  # greedy by default -> deterministic
    temperature: float | None = None  # only used if do_sample
    top_p: float | None = None  # only used if do_sample
    # Placeholder values: revisit after the pilot (truncation rate).
    max_new_tokens: int = 512


DECODING: dict[str, DecodingConfig] = {
    "gemma3-1b": DecodingConfig(),
    "gemma3-4b": DecodingConfig(),
    "gemma4-e4b": DecodingConfig(max_new_tokens=8192),  # long reasoning traces
}


# ---------------------------------------------------------------- secrets
def get_hf_token() -> str:
    """HF token via huggingface_hub's standard lookup. Never hardcode it.

    Checks, in order: Colab secret `HF_TOKEN`, env var `HF_TOKEN`, then the
    token saved by `huggingface-cli login`. Raises if none is found.
    """
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Log in with `huggingface-cli login`, "
            "set the HF_TOKEN env var, or add an HF_TOKEN Colab secret. "
            "See README §Setup (the Gemma 3 licenses must also be accepted)."
        )
    return token


# ---------------------------------------------------------------- run logging
def runtime_info() -> dict:
    """Hardware / library versions for the methodology section."""
    info: dict = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def run_dir(model_key: str, run_name: str = "baseline") -> Path:
    d = OUTPUTS_DIR / model_key / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_config(model_key: str, run_name: str = "baseline", **extra) -> Path:
    """Write config.json (model, decoding, seed, runtime, extras) into the run dir."""
    cfg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "model": asdict(MODELS[model_key]),
        "quantization": QUANTIZATION,
        "decoding": asdict(DECODING[model_key]),
        "runtime": runtime_info(),
        **extra,
    }
    path = run_dir(model_key, run_name) / "config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path
