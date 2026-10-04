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
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SPIDER_DIR = DATA_DIR / "spider"
OUTPUTS_DIR = REPO_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"  # Step 3: metric tables (CSV)
FIGURES_DIR = METRICS_DIR / "figures"  # Step 3: plots for the report
EVAL_SCRIPT_DIR = REPO_ROOT / "test-suite-sql-eval"
ARCHIVE_DIR = REPO_ROOT / "archive"  # local only (gitignored): pilot runs, superseded runs
PILOT_DIR = ARCHIVE_DIR / "pilot"  # pilot runs: archive/pilot/<model>/
PILOT_IDS = PILOT_DIR / "pilot_ids.json"
PROMPT_TEMPLATE = Path(__file__).parent / "prompt_template.txt"
DEV_INPUTS = Path(__file__).parent / "spider_dev_inputs.json"  # per example: id, db_id, question, gold query, difficulty, prompt

# ---------------------------------------------------------------- data sources
# Official Spider 1.0 zip (yale-lily.github.io/spider), includes the SQLite DBs
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
EVAL_SCRIPT_REPO = "https://github.com/taoyds/test-suite-sql-eval.git"
EVAL_SCRIPT_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"  # pinned for reproducibility

# ---------------------------------------------------------------- prompt
N_SAMPLE_ROWS = 3  # sample rows per table in the schema block
MAX_CELL_CHARS = 100  # truncate long cell values in sample rows

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
    loader: str  # transformers Auto class used to load it (per the model card)
    reasoning: bool = False  # True -> enable_thinking=True + THINKING_PREFILL
    gated: bool = False
    attn_implementation: str = "sdpa"  # see docs/inference_issues.md (Issue 1)

    @property
    def name(self) -> str:
        """HF repo name, e.g. gemma-3-1b-it (used in tables and reports)."""
        return self.hf_id.split("/")[-1]


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        # eager: SDPA's memory-efficient kernel corrupted some batched outputs (docs/inference_issues.md)
        ModelSpec(
            "gemma3-1b", "google/gemma-3-1b-it", "A", "~1.0B", "AutoModelForCausalLM",
            gated=True, attn_implementation="eager",
        ),
        ModelSpec(
            "gemma3-4b", "google/gemma-3-4b-it", "A", "~4.3B", "AutoModelForImageTextToText", gated=True
        ),
        ModelSpec(
            "gemma4-e4b",
            "google/gemma-4-E4B-it",
            "B",
            "~8.0B total / ~4.5B effective (PLE)",
            "AutoModelForMultimodalLM",
            reasoning=True,
        ),
    ]
}

# Same precision for all models: all three fit in bf16 on the A40 (46 GB).
QUANTIZATION = "bf16"  # one of: "bf16", "8bit", "4bit-nf4"


# ---------------------------------------------------------------- decoding
@dataclass
class DecodingConfig:
    do_sample: bool = False  # greedy by default -> deterministic
    temperature: float | None = None  # only used if do_sample
    top_p: float | None = None  # only used if do_sample
    top_k: int | None = None  # only used if do_sample
    # Same cap for all models; Gemma 3 stops at EOS long before it.
    max_new_tokens: int = 8192


# Identical settings for every model (greedy, same cap) to keep the comparison fair
DECODING: dict[str, DecodingConfig] = {key: DecodingConfig() for key in MODELS}
BATCH_SIZE = 16  # same for every model

# E4B decides by itself whether to think, and skipped thinking on easy questions
# in a probe. Pre-filling the model turn with the thought-channel opener forces a
# trace for every example (Task 2 needs one per example).
THINKING_PREFILL = "<|channel>thought\n"


# ---------------------------------------------------------------- secrets
def get_hf_token() -> str:
    """HF token via huggingface_hub's standard lookup. Never hardcode it.

    Checks, in order: env var `HF_TOKEN`, then the token saved by
    `huggingface-cli login`. Raises if none is found.
    """
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Log in with `huggingface-cli login`, "
            "or set the HF_TOKEN env var. "
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


def run_path(model_key: str, run_name: str = "baseline") -> Path:
    """Where a run's files live: archive/pilot/<model>/ for the pilot, outputs/<model>/<run>/ otherwise."""
    if run_name == "pilot":
        return PILOT_DIR / model_key
    return OUTPUTS_DIR / model_key / run_name


def run_dir(model_key: str, run_name: str = "baseline") -> Path:
    """run_path(), created if missing."""
    d = run_path(model_key, run_name)
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
