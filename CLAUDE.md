# CISC 839 – Assignment 1: LLMs for Text-to-SQL

## Goal
Evaluate, analyze, and improve LLMs on **Text-to-SQL** (natural-language question → SQL):
1. Benchmark LLMs of different sizes/types (Task 1)
2. Analyze reasoning traces of a reasoning LLM vs. performance (Task 2)
3. Improve one solution, via better reasoning or parameter adaptation (Task 3)
4. AI usage declaration (Task 4, report only)

Running LLMs on hundreds of examples (especially reasoning models) can take hours. Prefer resumable runs that cache outputs to disk.

## Dataset: Spider 1.0
- Source: https://github.com/taoyds/spider. Each example has an NL question, a SQLite DB schema, and a gold SQL query.
- **Train split**: ONLY for few-shot example selection (Task 1/3) or fine-tuning (Task 3).
- **Dev split**: ALL evaluation. Never use dev for few-shot examples or training.
- An evaluation subset of dev is allowed. The sampling must be meaningful (e.g., stratified by difficulty), use a fixed random seed, and **record the list of example IDs used**.
- Metrics: follow the Spider paper by default (Exact Match, Execution Accuracy).
- Difficulty levels (easy/medium/hard/extra) come from the official eval script: https://github.com/taoyds/test-suite-sql-eval/blob/master/evaluation.py

## Task 1: Benchmark (RQ1: How do selected LLMs perform on cross-domain Text-to-SQL?)
Evaluate **≥3 models** covering:
- **A. Decoder-only, no explicit reasoning**: ≥2 models of *different sizes from the same family* (e.g., Qwen2.5 / Qwen2.5-Coder)
- **B. Reasoning LLM**: ≥1 model that outputs an explicit reasoning trace (e.g., DeepSeek-R1-Distill-Qwen, Qwen3+ with thinking mode)

Open-weight (Hugging Face) or API models are both allowed.

Must capture and report, in this order:
- **Models**: name, version/checkpoint, param count (total + active for MoE), quantization, access method (local/API)
- **Eval data**: split, subset sampling method, random seed
- **Prompting**: prompt template, schema representation, few-shot examples, SQL extraction method
- **Inference**: decoding strategy, temperature, top-p, max output length, reasoning on/off, hardware/runtime
- **Evaluation**: metrics, how invalid/unparseable outputs are counted

Results: summary table of all models, overall and per difficulty level, plus efficiency measures (inference time and output tokens per example). Present results first, then interpret them with specific numbers.

## Chosen models (Task 1)
Hugging Face IDs checked against the HF API on 2026-10-02.

| Category | Model | HF checkpoint | Params (counted from weights) | Notes |
|---|---|---|---|---|
| A. Decoder-only, no reasoning | Gemma 3 1B IT | `google/gemma-3-1b-it` @ `dcc83ea` | 1.00B | Text-only; **gated** |
| A. Decoder-only, no reasoning | Gemma 3 4B IT | `google/gemma-3-4b-it` @ `093f9f3` | 4.30B (3.88B text + 0.42B vision) | Multimodal (image-text-to-text); use for text only; **gated** |
| B. Reasoning LLM | Gemma 4 E4B IT | `google/gemma-4-E4B-it` @ `ee0ef60` | 8.00B total / 4.51B effective | Dense + Per-Layer Embeddings (not MoE); multimodal (any-to-any); not gated |

Revisions are pinned in `src/config.py` (`ModelSpec.revision`, full hashes) and loaded by `task_1.ipynb` Step 2.2. They were verified on 2026-10-06 as the only cached snapshots, so they are the weights behind all reported runs. Full model details for the methodology: `archive/docs/task_1_methodology.txt` (local, gitignored).

Model notes:
- **Gated Gemma 3 models**: accept the license on each HF model page and authenticate with an HF token. Keep the token in the `HF_TOKEN` environment variable (or `huggingface-cli login`), never in code.
- **E4B thinking mode must be ON**: add `<|think|>` at the start of the system prompt (check against the model card or chat template for the inference setup used). Confirm the reasoning trace appears in the output and is **saved in full per example**, because Task 2 depends on it.
- **Report both param counts for E4B** (8B total / ~4.5B effective) and explain "effective" (PLE) in the methodology. The spec asks for total + active counts for MoE models, and the same reasoning applies here.
- **Long traces**: set a generous max output length for E4B and decide up front how to count truncated or unparseable outputs (default: count as incorrect/invalid, and report how many there were).
- **Generation confound**: the reasoning model is Gemma 4 and the non-reasoning ones are Gemma 3. Gaps between the groups mix reasoning with generational improvements, so mention this in the RQ1 interpretation. E4B (~4.5B effective) vs Gemma 3 4B is the closest size-matched comparison.
- **Memory**: E4B in bf16 needs ~16 GB for weights alone. This fits on the container's A40 (46 GB), so all three models run in bf16.
- **Quantization**: record the precision for every model in the methodology (e.g., "bf16" or "4-bit NF4 via bitsandbytes"). Use the same precision for all models where possible; otherwise note the difference as a limitation.
- **Multimodal loading**: Gemma 3 4B and Gemma 4 E4B include vision/audio encoders, so load them with the correct model class (check the model cards). Only text input is used.
- **Pilot first**: before full runs, run each model on ~20–30 dev examples to measure seconds/example, output tokens, and truncation rate.

## Task 2: Reasoning traces (RQ2: How do trace characteristics relate to performance?)
- Use the reasoning traces collected from the Task 1 reasoning model, so **save full raw traces** in Task 1.
- Define **≥2 measurable features** per trace (definition, relevance to Text-to-SQL, how it's computed).
- Name the statistical tests/models relating each feature to correctness and difficulty, and justify why each fits the data.
- Results: descriptive stats by correctness × difficulty, statistical results, and ≥1 visualization (e.g., box plots).
- **Task 2 plan (2026-10-06):**
    - Traces: Gemma 4 E4B (the only model with traces), field `thinking` in `outputs/gemma4-e4b/baseline/responses.jsonl`. Correctness (EX, EM) and difficulty come from `outputs/task1/metrics/task1_per_example.csv`, joined by `id`.
    - Features chosen with the user: **(1) trace length** (thinking tokens) and **(2) number of SQL drafts** written out in the trace. **Draft definition (A, decided 2026-10-06):**
        - a draft is every distinct SQL query written out in the trace that contains both `SELECT` and `FROM`;
        - it can be in a code block, in inline code, or written clause by clause on consecutive lines (joined into one query);
        - queries are compared after normalizing whitespace and case, so repeats count once;
        - sub-queries and rewrites count as separate drafts;
        - keyword mentions (e.g. `COUNT(*)`) are not drafts.
        - On the 1034 traces: 0 drafts in 57, 1 in 665, ≥2 in 312.
        - Rejected alternative (B): drop sub-queries that appear inside a later query. Its heuristic missed some building blocks.
    - Results must include, as requested by the user:
        - descriptive statistics for each feature, broken down by correctness and difficulty;
        - the statistical test results;
        - at least one visualization, e.g. box plots of each feature by correctness and difficulty.
    - Every feature co-varies with difficulty, so the tests must control for difficulty (compare correct vs. wrong within each level).
    - **UPDATE (2026-10-06): Task 2 analysis done** in `src/task_2.ipynb`.
        - Correctness = EX.
        - Tests (chosen with the user): Kruskal–Wallis + Spearman (feature vs difficulty); Mann–Whitney U + rank-biserial r within each level, Holm over 8 tests; logistic regression correct ~ feature + difficulty (length as log2 tokens).
        - Plots: box plots by difficulty × correctness with means marked, plus accuracy by number of drafts.
        - Outputs: `outputs/task2/metrics/task2_{features,descriptive,tests,logistic}.csv`, `outputs/task2/figures/task2_{feature_boxplots,accuracy_by_drafts}`.
        - The draft counter was hand-checked on 24 traces (24/24 agree). In plain text, only UPPERCASE SQL keywords count, because E4B uses 'Select … from' as ordinary English.
        - Notebook markdown explains why each test was chosen and what it is applied to, and the code has step comments (user requirement).
        - Next: Task 2 write-up (`archive/docs/task_2_*.txt`; cite Kruskal & Wallis 1952, Spearman 1904, Mann & Whitney 1947 and Holm 1979, verified first).

## Task 3: Improve one solution (RQ3: To what extent does [strategy] improve [model]?)
Choose ONE:
- **Option A – Improving reasoning** (e.g., prompting/reasoning strategy)
- **Option B – Adaptation**: change model parameters using the Spider **train** split (e.g., fine-tuning/LoRA)

Justify the choice of model, name the strategy, and evaluate on the **same subset and metrics as Task 1**, overall and by difficulty. Include one table comparing original vs. improved, plus 1–2 examples the change fixed or broke. A negative result is acceptable if it is well motivated and explained.

## Task 4: AI usage declaration
One paragraph in the report covering the tools used (including Claude Code), what they helped with, and how their output was checked.
- **Must mention (noted 2026-10-06):** Claude Code drafted the report content for Task 1 (`archive/docs/task_1_methodology.txt`, `task_1_results.txt`, generated from the result CSVs) and the figures. The user edits and verifies everything manually before submission. Also cover the code, analysis and debugging help, and how it was checked (official-script cross-checks, reproducibility reruns, verified citations).

## Deliverables
1. **Code**: notebook(s)/scripts that reproduce Tasks 1–3, plus a **README** with setup instructions. The TA may rerun the code on the eval subset.
   - Do NOT commit/submit model weights.
   - **Never hardcode or commit API tokens.** Use environment variables.
2. **Report (PDF)** with exactly these headings:
   1. Task 1 – RQ1: Methodology; Results
   2. Task 2 – RQ2: Methodology; Results
   3. Task 3 – RQ3: Methodology; Results
   4. Task 4 – AI Usage Declaration

   Every table and figure needs a caption and must be referenced in the text. Everything graded must be in the report, because TAs won't search the code.

## Runtime environment: local container (NVIDIA A40, 46 GB)
- All work (data prep, inference, any fine-tuning) runs in **this container**: NVIDIA A40 46 GB, torch 2.11.0+cu126, conda Python 3.12. Google Colab is **not used at all** (decided 2026-10-03; replaces the 2026-10-02 Colab decision). Don't add Colab-specific code.
- **README** (done 2026-10-03): `README.md` covers the CUDA/≥24 GB VRAM requirement, install, the gated Gemma 3 license + HF token steps (`hf auth login` / `HF_TOKEN`, with an access check), and states that results were produced on an A40. Keep it updated as Tasks 2–4 land.

## Repo state / conventions
- Notebooks live in `src/` (`task_1.ipynb`, `task_2.ipynb`, `task_3.ipynb`). Each starts with a `%pip install` cell and a setup cell. Shared settings are in `src/config.py` (paths, seed, model IDs, decoding params and run logging), next to `src/prompt_template.txt` and the model inputs `src/spider_dev_inputs.json`; each notebook imports it with `import config as C`. Task logic (data prep, prompting, inference, metrics) lives in the task notebook itself as separate cells, each with a markdown header and description; `src/config.py` holds only shared settings/helpers. There is no `src/text2sql/` package any more (removed 2026-10-04).
- Log all settings (seeds, model IDs, decoding params) and save per-example outputs (prediction, raw output, trace, latency, token counts, correctness, difficulty) to files so report tables can be regenerated.


## CURRENT TASKS
1. 
- before any coding: claude, read through the dev dataset and create a plain text prompt template that will be used for prompting the models. 
- figure out how to represent the database schema for each example in the dev dataset. this should be included in the prompt, as extra context. 
- also tell each LLM how to respond. we want the response to be JUST the SQL query, so that we can compare them against the ground-truth queries.
- we will confirm/settle on a good prompt template.

2. 
- download spider dataset (DEV one)
- iterate through each element in the .json dataset (i.e. each text-SQL pair), and add an additional key: “prompt”... i.e. this is on the same level as “question” or “db_id”. at the same time that you add this new key, add the prompt according to the template made, and this will be the respective value to each individual entry’s “prompt”. the data needed to fill-in the template will be pulled from other keys in the same data point entry, such as the “question” and other information about the database schema. 
- once this is complete, we will be adding another new key called “difficulty”, whose value will hold the results of this script for each respective data point: https://github.com/taoyds/test-suite-sql-eval/blob/master/evaluation.py
- we will be adding another new key called “responses”… the value of this will be 3 sub-dictionaries. the key of each sub-dict will be model’s name, e.g. gemma-4-E4B-it. the value for each will be the LLM’s actual response that is output…
- **UPDATE (2026-10-04), storage decision:** the responses are NOT stored inside the dataset. Merging them into a copy of the dev set duplicated what the per-model output files already hold, so we dropped it.
    - Model inputs: `src/spider_dev_inputs.json`, one entry per dev example with `id` (position in Spider `dev.json`), `db_id`, `question`, gold `query`, `difficulty`, `prompt`.
    - Model outputs: only in `outputs/<model>/baseline/responses.jsonl`, one record per example keyed by the same `id` (extracted SQL in `query`, plus raw output, token counts, latency). Gemma 4 E4B's reasoning trace is the `thinking` field there (no separate `reasoning_traces` key).
    - Metrics and Task 2 join inputs and outputs by `id`.

3. 
- we will be testing on ALL the dev examples…
- we will are doing zero-shot sampling
- we want to consider the following before we actually start prompting, and want to make sure that it is as consistent as possible across all 3 LLMs to reduce bias when comparing metrics…. 
    - Inference settings: decoding strategy, temperature, top-p, maximum output length, whether reasoning was enabled, and the hardware or runtime used
- now ready for prompting the LLMs. prompt each model and save its output in its associated place under “responses” (UPDATE 2026-10-04: done; outputs live in `outputs/<model>/baseline/responses.jsonl`, see part 2). again, the “<model-name>” and “query” should be in the exact same format in order to compare and grab statistics.

- repeat with all 3 models. now, we want to grab metrics. we will stop and have claude grab the metrics that were used in the Spider dataset’s associated paper: https://arxiv.org/pdf/1809.08887

4. 
- **Start of the metrics step (moved here from the removed task_1 Step 2.7, 2026-10-04):** load `src/spider_dev_inputs.json` and every `outputs/<model>/baseline/responses.jsonl`, join them by `id`, and assert that each model's output ids are exactly the input ids and that Gemma 4 E4B has a non-empty `thinking` trace for every example (Task 2 needs them).
- then, we will create a new file to store these metrics.
- Report performances of all considered models, overall and by difficulty level, in a summary table. perhaps we make a .xlsx or .csv file. these metrics will also definitely be plotted for the report.
- compare the accuracy of each LLM to the ground-truth query
- compare the accuracy of each LLM to each other
- how will we count invalid or unparseable outputs?
- Also report efficiency measures (e.g., inference time and output tokens per example).
- **UPDATE (2026-10-04): part 4 done** in `task_1.ipynb` Step 3. Decisions:
    - Metrics: EM + component F1 (Spider paper) + EX (official script; not reported in the paper).
    - Supplementary "EM (normalized)": `INNER JOIN` → `JOIN` before scoring. It is our addition, because the 2018 parser can't read `INNER JOIN`, which mainly affects E4B. Explained in `archive/docs/task_1_methodology.txt` (local).
    - Model-vs-model comparison: descriptive only (2×2 overlap counts and accuracy differences, no significance tests).
    - Invalid, unparseable and exec-error outputs count as incorrect and are reported separately.
    - Results: CSVs in `outputs/task1/metrics/`, figures in `outputs/task1/figures/`.
    - **Central results table: `outputs/task1/metrics/task1_summary.csv`**, one row per model: EX/EM by difficulty and overall, normalized EM, s/example, output tokens, thinking tokens. It is the report's main table.
    - Supporting tables: `task1_accuracy_by_difficulty.csv`, `task1_efficiency.csv`, `task1_efficiency_by_difficulty.csv` (added 2026-10-04: time and tokens per difficulty level), `task1_component_f1.csv`, `task1_invalid_outputs.csv`, `task1_model_agreement.csv`, `task1_per_example.csv`.
    - Explanation of all metrics: `archive/docs/task_1_methodology.txt` (local).
- **UPDATE (2026-10-06): Task 1 write-up.**
    - The four `docs/*.md` notes were consolidated into two paste-ready LaTeX files for the Overleaf report: `archive/docs/task_1_methodology.txt` (Models → Eval data → Prompting → Inference → Evaluation, with BibTeX entries at the end) and `archive/docs/task_1_results.txt` (tables, figures, interpretation, answer to RQ1). Both are in the gitignored `archive/docs/` (not committed; decided 2026-10-06).
    - All numbers in them were generated from `outputs/task1/metrics/*.csv` by a script kept locally in `archive/writeup/` (gitignored; rerun `python archive/writeup/build_writeup.py` if the metrics change; it writes to `archive/docs/`).
    - Figures (5, each as PNG + PDF, made in `task_1.ipynb` Step 3.6): `task1_accuracy_by_difficulty`, `task1_component_f1`, `task1_model_agreement`, `task1_errors`, `task1_efficiency`.
    - Per-example scores were cross-checked to equal the official `evaluation.py` output.