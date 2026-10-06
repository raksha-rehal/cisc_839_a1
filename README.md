# CISC 839 – Assignment 1: LLMs for Text-to-SQL

This repo evaluates three open-weight LLMs on Spider 1.0 (dev, all 1034 examples) for Text-to-SQL, using zero-shot prompting:

| Model | HF checkpoint | Pinned revision | Parameters | Role | Gated |
|---|---|---|---|---|---|
| Gemma 3 1B IT | `google/gemma-3-1b-it` | `dcc83ea` | 1.00B | decoder-only, no reasoning | **yes** |
| Gemma 3 4B IT | `google/gemma-3-4b-it` | `093f9f3` | 4.30B (3.88B text) | decoder-only, no reasoning | **yes** |
| Gemma 4 E4B IT | `google/gemma-4-E4B-it` | `ee0ef60` | 8.00B total / 4.51B effective | reasoning (thinking mode) | no |

The code loads each model at the pinned revision (full hashes in `src/config.py`), so reruns use exactly the weights behind our results.

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

`requirements.txt` **pins the exact package versions** that produced our results (e.g. `transformers==5.18.0`), so a rerun uses the same software. Please don't upgrade them; newer versions can change model outputs slightly.

**PyTorch and your GPU:** PyTorch builds are tied to a CUDA version. Our results used `torch==2.11.0` built for CUDA 12.6 (`2.11.0+cu126`, NVIDIA driver 580). After installing, check that PyTorch sees the GPU:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If it prints `False`, keep version 2.11.0 but reinstall the build that matches your driver, e.g. `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126` (replace `cu126` with the CUDA version your driver supports; see pytorch.org).

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
  - **Step 3** scores every prediction with the official Spider evaluation code: Execution Accuracy (EX), Exact Match (EM, official and normalized `INNER JOIN` → `JOIN`) and component F1, overall and by difficulty. It also covers model-vs-model agreement, invalid-output counts and efficiency, overall and by difficulty. Tables are written to `outputs/task1/metrics/*.csv` (the central results table is **`outputs/task1/metrics/task1_summary.csv`**) and figures to `outputs/task1/figures/`. Our scores are checked against the official script's printed output. Each metric, and a limitation of official EM, is explained in the notebook's Step 3 text. Step 3 needs no GPU and takes ~5 min.
- **`src/task_2.ipynb`** (Task 2, reasoning traces): measures two features of every Gemma 4 E4B reasoning trace: trace length (thinking tokens) and the number of SQL drafts written in the trace. It relates them to correctness (EX) and difficulty with descriptive statistics, Kruskal–Wallis and Spearman tests, within-difficulty Mann–Whitney U tests (Holm-corrected), and logistic regression controlling for difficulty. It reads only Task 1's committed outputs, so it needs no GPU and runs in seconds. Tables: `outputs/task2/metrics/task2_*.csv`; figures: `outputs/task2/figures/task2_*`.
- **`src/task_3.ipynb`** (Task 3, fine-tuning): fully fine-tunes Gemma 3 1B on Spider's `train_spider.json` with the same prompt format as Task 1. It holds out about 10% of the training databases (seed 42) as a validation set, recorded in `outputs/task3/splits.json`. After every epoch it measures validation EX and keeps the best epoch, then evaluates that model on all 1034 dev examples exactly as in Task 1 (run `outputs/gemma3-1b/finetuned/`). Training takes about 1.5 h on an A40 and needs about 38 GB of GPU memory. Model weights are saved to the gitignored `checkpoints/` folder and are **not** in the repo, so rerunning the notebook retrains the model; GPU training is not bit-for-bit deterministic, so the retrained weights and scores can differ slightly. Steps 3.2–3.4 (scoring, comparison and figures) read the committed outputs. Tables: `outputs/task3/metrics/task3_*.csv`; figures: `outputs/task3/figures/task3_*`.

All settings (seed, model IDs, decoding, batch size, fine-tuning settings, paths) are in `src/config.py`. Code shared by Tasks 1 and 3 (schema rendering, prompts, generation, SQL extraction, scoring) is in `src/text2sql_utils.py`. The prompt template is `src/prompt_template.txt`. The model inputs built in Step 1 (prompt, difficulty and gold SQL per example) are in `src/spider_dev_inputs.json`.
