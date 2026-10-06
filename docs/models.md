# Models (Task 1)

This file covers the "Models" part of the Task 1 methodology: name, version/checkpoint, parameter counts, precision and access method. All values were checked against the weights actually used (2026-10-06).

## Summary

| | Gemma 3 1B IT | Gemma 3 4B IT | Gemma 4 E4B IT |
|---|---|---|---|
| Role (Task 1 category) | A: decoder-only, no reasoning | A: decoder-only, no reasoning | B: reasoning (thinking mode, forced trace) |
| Hugging Face checkpoint | `google/gemma-3-1b-it` | `google/gemma-3-4b-it` | `google/gemma-4-E4B-it` |
| **Revision (commit) used** | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | `093f9f388b31de276ce2de164bdc2081324b9767` | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Total parameters | 1.00B (999,885,952) | 4.30B (4,300,079,472) | 8.00B (7,996,157,418) |
| Text model only | 1.00B | 3.88B | 7.52B |
| Unused modality encoders | none | vision 0.42B | vision 0.17B, audio 0.31B |
| **Effective parameters** | n/a | n/a | **4.51B** (see below) |
| Architecture | dense decoder-only, text-only | dense decoder-only + vision encoder | dense decoder-only + Per-Layer Embeddings (PLE) + vision/audio encoders (**not** MoE) |
| Precision | bf16 (no quantization) | bf16 | bf16 |
| Attention implementation | eager (see `docs/inference_issues.md`) | PyTorch SDPA | PyTorch SDPA |
| Loaded with (transformers) | `AutoModelForCausalLM` | `AutoModelForImageTextToText` | `AutoModelForMultimodalLM` |
| Access | open weights, run locally | open weights, run locally | open weights, run locally |
| Gated on Hugging Face | yes (license + token) | yes (license + token) | no |

**Access method:** all three are open-weight models, downloaded from Hugging Face and run **locally** with Hugging Face `transformers` 5.18.0 (torch 2.11.0+cu126) on one NVIDIA A40 (46 GB). No API is used. Only text input is used, even for the two multimodal models.

## Versions: pinned revisions
A Hugging Face model repository can be updated after we used it, so the model name alone doesn't identify the weights. Each model is therefore pinned to the exact repository commit (revision) we used: `ModelSpec.revision` in `src/config.py`. `task_1.ipynb` Step 2.2 loads that revision, so a rerun gets byte-identical weights.

How we know these are the revisions behind every reported run:
- The local Hugging Face cache holds **exactly one snapshot** per model, and these hashes are those snapshots.
- On 2026-10-06, Hugging Face's `main` branch still pointed at these same commits for all three repositories.

The run logs written before the revisions were pinned (`outputs/<model>/baseline/config.json`) record the model name but not the revision. This file is the record for those runs. Runs made after the pin also log `revision` in their `config.json`.

## Parameter counts and what "effective" means
- **How we counted:** we read the tensor shapes in each model's weight files (`*.safetensors`) and summed them. No weights were loaded. The totals match the counts Hugging Face publishes for each repository. For E4B the published figure is 7,996,156,490, which differs from our count by 928 parameters out of 8 billion, a negligible amount. All weights are stored in bf16.
- **Gemma 3 4B:** its 4.30B total includes a 0.42B vision encoder that a text-only prompt never uses. The language model itself is 3.88B.
- **Gemma 4 E4B, "effective" parameters:** the "E" in E4B stands for *effective*. Besides normal token embeddings, E4B has **Per-Layer Embeddings (PLE)**: each of its 42 decoder layers gets its own small embedding for every token. These tables are big (2.82B parameters), but they are only *looked up*, one row per token, not multiplied through like the rest of the network. So they add little computation. The model card describes E4B as "4.5B effective (8B with embeddings)". Our counts reproduce that:

  | Component | Parameters |
  |---|---|
  | Total | 7.996B |
  | − Per-layer embeddings (PLE) | 2.819B |
  | − Token embeddings | 0.671B |
  | **= Effective** | **4.506B** |

  The remaining 4.51B are the transformer layers plus the vision and audio encoders. We report **both** numbers, "8.0B total / ~4.5B effective". This is similar to how a Mixture-of-Experts model is reported as total / active parameters, but E4B is **not** an MoE model: every layer runs for every token, and only the embedding lookups are cheap.

## Implications for comparing the models
- **Size-matched pair:** Gemma 4 E4B (~4.5B effective) vs Gemma 3 4B (4.30B total, 3.88B text model) is the closest size-matched comparison between the reasoning and non-reasoning groups.
- **Generation confound:** the reasoning model is Gemma 4 and the non-reasoning models are Gemma 3, so differences between the groups combine the effect of reasoning with improvements from the newer model generation. The RQ1 interpretation must mention this.

## Sources
- Weight files and repository revisions: Hugging Face Hub, via the local cache and the Hub API (`/api/models/<id>/revision/main`), checked 2026-10-06.
- Gemma 4 E4B model card (at the pinned revision): the "Total Parameters: 4.5B effective (8B with embeddings)" table and the explanation of "effective" parameters and Per-Layer Embeddings.
