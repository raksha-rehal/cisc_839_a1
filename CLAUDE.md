# CISC 839 – Assignment 1: LLMs for Text-to-SQL

## Goal
Evaluate, analyze, and improve LLMs on **Text-to-SQL** (natural-language question → SQL):
1. Benchmark LLMs of different sizes/types (Task 1)
2. Analyze reasoning traces of a reasoning LLM vs. performance (Task 2)
3. Improve one solution, via better reasoning or parameter adaptation (Task 3)
4. AI usage declaration (Task 4, report only)

Running LLMs on hundreds of examples (especially reasoning models) can take hours, and free GPU sessions such as Colab have time limits. Prefer resumable runs that cache outputs to disk.

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

Open-weight (Hugging Face / Colab) or API models are both allowed.

Must capture and report, in this order:
- **Models**: name, version/checkpoint, param count (total + active for MoE), quantization, access method (local/API)
- **Eval data**: split, subset sampling method, random seed
- **Prompting**: prompt template, schema representation, few-shot examples, SQL extraction method
- **Inference**: decoding strategy, temperature, top-p, max output length, reasoning on/off, hardware/runtime
- **Evaluation**: metrics, how invalid/unparseable outputs are counted

Results: summary table of all models, overall and per difficulty level, plus efficiency measures (inference time and output tokens per example). Present results first, then interpret them with specific numbers.

## Chosen models (Task 1)
Hugging Face IDs checked against the HF API on 2026-10-02.

| Category | Model | HF checkpoint | Params (safetensors) | Notes |
|---|---|---|---|---|
| A. Decoder-only, no reasoning | Gemma 3 1B IT | `google/gemma-3-1b-it` | ~1.0B | Text-only; **gated** |
| A. Decoder-only, no reasoning | Gemma 3 4B IT | `google/gemma-3-4b-it` | ~4.3B | Multimodal (image-text-to-text); use for text only; **gated** |
| B. Reasoning LLM | Gemma 4 E4B IT | `google/gemma-4-E4B-it` | ~8.0B total / ~4.5B effective | Dense + Per-Layer Embeddings (not MoE); multimodal (any-to-any); not gated |

Model notes:
- **Gated Gemma 3 models**: accept the license on each HF model page and authenticate with an HF token. Keep the token in an environment variable or Colab secret, never in code.
- **E4B thinking mode must be ON**: add `<|think|>` at the start of the system prompt (check against the model card or chat template for the inference setup used). Confirm the reasoning trace appears in the output and is **saved in full per example**, because Task 2 depends on it.
- **Report both param counts for E4B** (8B total / ~4.5B effective) and explain "effective" (PLE) in the methodology. The spec asks for total + active counts for MoE models, and the same reasoning applies here.
- **Long traces**: set a generous max output length for E4B and decide up front how to count truncated or unparseable outputs (default: count as incorrect/invalid, and report how many there were).
- **Generation confound**: the reasoning model is Gemma 4 and the non-reasoning ones are Gemma 3. Gaps between the groups mix reasoning with generational improvements, so mention this in the RQ1 interpretation. E4B (~4.5B effective) vs Gemma 3 4B is the closest size-matched comparison.
- **Memory**: E4B in bf16 needs ~16 GB for weights alone, which won't fit on a free Colab T4. Use 8-bit/4-bit quantization or an L4/A100.
- **Quantization**: record the precision for every model in the methodology (e.g., "bf16" or "4-bit NF4 via bitsandbytes"). Use the same precision for all models where possible; otherwise note the difference as a limitation.
- **Multimodal loading**: Gemma 3 4B and Gemma 4 E4B include vision/audio encoders, so load them with the correct model class (check the model cards). Only text input is used.
- **Pilot first**: before full runs, run each model on ~20–30 dev examples to measure seconds/example, output tokens, and truncation rate.

## Task 2: Reasoning traces (RQ2: How do trace characteristics relate to performance?)
- Use the reasoning traces collected from the Task 1 reasoning model, so **save full raw traces** in Task 1.
- Define **≥2 measurable features** per trace (definition, relevance to Text-to-SQL, how it's computed).
- Name the statistical tests/models relating each feature to correctness and difficulty, and justify why each fits the data.
- Results: descriptive stats by correctness × difficulty, statistical results, and ≥1 visualization (e.g., box plots).

## Task 3: Improve one solution (RQ3: To what extent does [strategy] improve [model]?)
Choose ONE:
- **Option A – Improving reasoning** (e.g., prompting/reasoning strategy)
- **Option B – Adaptation**: change model parameters using the Spider **train** split (e.g., fine-tuning/LoRA)

Justify the choice of model, name the strategy, and evaluate on the **same subset and metrics as Task 1**, overall and by difficulty. Include one table comparing original vs. improved, plus 1–2 examples the change fixed or broke. A negative result is acceptable if it is well motivated and explained.

## Task 4: AI usage declaration
One paragraph in the report covering the tools used (including Claude Code), what they helped with, and how their output was checked.

## Deliverables
1. **Code**: notebook(s)/scripts that reproduce Tasks 1–3, plus a **README** with setup instructions. The TA may rerun the code on the eval subset.
   - Do NOT commit/submit model weights.
   - **Never hardcode or commit API tokens.** Use environment variables or Colab secrets.
2. **Report (PDF)** with exactly these headings:
   1. Task 1 – RQ1: Methodology; Results
   2. Task 2 – RQ2: Methodology; Results
   3. Task 3 – RQ3: Methodology; Results
   4. Task 4 – AI Usage Declaration

   Every table and figure needs a caption and must be referenced in the text. Everything graded must be in the report, because TAs won't search the code.

## Runtime environment: Google Colab
- All model runs (inference, any fine-tuning) happen on **Google Colab** (decided 2026-10-02). Target Colab's GPUs and its preinstalled `torch`; don't assume a local GPU or a university cluster.
- **README TODO**: tell the TA to rerun the notebooks on **Google Colab** (GPU runtime). Include the Colab steps: open/clone the repo, add the `HF_TOKEN` secret, accept the Gemma 3 licenses, and state which GPU type the results were produced on.

## Repo state / conventions
- Notebooks live in `src/` (`task_1.ipynb`, `task_2.ipynb`, `task_3.ipynb`). Each starts with a `%pip install` cell and a setup cell. Shared code is in `src/text2sql/` (`config.py` holds paths, seed, model IDs, decoding params and run logging).
- Log all settings (seeds, model IDs, decoding params) and save per-example outputs (prediction, raw output, trace, latency, token counts, correctness, difficulty) to files so report tables can be regenerated.
