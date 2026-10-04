# CISC 839 – Assignment 1: LLMs for Text-to-SQL

This repo evaluates three open-weight LLMs on Spider 1.0 (dev, all 1034 examples) for Text-to-SQL, using zero-shot prompting:

| Model | HF checkpoint | Role | Gated |
|---|---|---|---|
| Gemma 3 1B IT | `google/gemma-3-1b-it` | decoder-only, no reasoning | **yes** |
| Gemma 3 4B IT | `google/gemma-3-4b-it` | decoder-only, no reasoning | **yes** |
| Gemma 4 E4B IT | `google/gemma-4-E4B-it` | reasoning (thinking mode) | no |

## Setup

### 1. Hardware / software
- A CUDA GPU with **≥ 24 GB VRAM**. All models run in bf16, and Gemma 4 E4B peaked at ~19.4 GB in the pilot. **Our results were produced on one NVIDIA A40 (46 GB)** with Python 3.12, torch 2.11.0+cu126 and transformers 5.18.0.
- About 30 GB of free disk space: ~27 GB for the model weights in the Hugging Face cache and ~2 GB for the Spider data.

### 2. Clone and install
```bash
git clone https://github.com/raksha-rehal/cisc_839_a1.git
cd cisc_839_a1
pip install -r requirements.txt
```
Each notebook also starts with a `%pip install -r ../requirements.txt` cell.

### 3. Hugging Face access to the gated Gemma 3 models (required)
The two **Gemma 3** models are **gated**: Hugging Face only lets you download them after you accept Google's Gemma license **with your own account**. No token is included in this repo, and none ever should be. Without these steps, loading Gemma 3 fails with a `401` / `GatedRepoError`.

1. Log in at https://huggingface.co (or create a free account).
2. Open **both** model pages and accept the license on each. Access is usually granted immediately.
   - https://huggingface.co/google/gemma-3-1b-it
   - https://huggingface.co/google/gemma-3-4b-it
3. Create an access token with **Read** permission at https://huggingface.co/settings/tokens.
4. Make the token available on the machine that runs the notebooks, using **one** of these:
   ```bash
   hf auth login            # paste the token when prompted (stored in ~/.cache/huggingface)
   # or
   export HF_TOKEN=hf_...   # environment variable, in the shell that starts Jupyter
   ```
5. Check your access. This should print `access OK` twice:
   ```bash
   python -c "
   from huggingface_hub import hf_hub_download
   for m in ['google/gemma-3-1b-it', 'google/gemma-3-4b-it']:
       hf_hub_download(m, 'config.json'); print(m, 'access OK')"
   ```

Gemma 4 E4B is **not** gated and downloads without a token. The code reads the token only through `huggingface_hub` (`src/config.py::get_hf_token`), never from the code itself.

## Running

Start Jupyter from the repo, open the notebooks in `src/`, and run them top to bottom:

- **`src/task_1.ipynb`** (Task 1, benchmark)
  - **Step 1** downloads Spider 1.0 (official zip, including the SQLite databases) to `data/spider/` and the official evaluation script (`taoyds/test-suite-sql-eval`, pinned commit) to `test-suite-sql-eval/`. It then builds `src/spider_dev_inputs.json`, which holds every dev example's `id`, `db_id`, `question`, gold `query`, official `difficulty` and `prompt`. No GPU is needed for this step.
  - **Step 2** runs a 30-example pilot, then all 1034 examples for each model. Each model's outputs are stored only in `outputs/<model>/<run>/responses.jsonl`: one record per example, matched to the inputs by `id`, holding the extracted SQL (`query`), raw output, token counts and latency. For Gemma 4 E4B the record also holds the full reasoning trace (`thinking`). Runs are resumable: records are appended after every batch, so if a run is interrupted, just rerun the cell. Each run's settings are logged to `outputs/<model>/<run>/config.json`. The pilot (Step 2.5) is only a local sanity check. Its outputs go to the gitignored `archive/pilot/` (`pilot_ids.json` plus one folder per model) and are not committed. Rerunning Step 2.5 regenerates them, with the same ids and outputs.
  - **Step 3** scores every prediction with the official Spider evaluation code: Execution Accuracy (EX), Exact Match (EM, official and normalized `INNER JOIN` → `JOIN`) and component F1, overall and by difficulty. It also covers model-vs-model agreement, invalid-output counts and efficiency, overall and by difficulty. Tables are written to `outputs/metrics/*.csv` (the central results table is **`outputs/metrics/task1_summary.csv`**) and figures to `outputs/metrics/figures/`. Our scores are checked against the official script's printed output. The metrics, and a limitation of official EM, are explained in `docs/evaluation_metrics.md`. Step 3 needs no GPU and takes ~5 min.
- `src/task_2.ipynb`, `src/task_3.ipynb`: Tasks 2 and 3 (in progress).

All settings (seed, model IDs, decoding, batch size, paths) are in `src/config.py`. The prompt template is `src/prompt_template.txt`, and the design is explained in `docs/prompt_design.md`. The model inputs built in Step 1 (prompt, difficulty and gold SQL per example) are in `src/spider_dev_inputs.json`.
