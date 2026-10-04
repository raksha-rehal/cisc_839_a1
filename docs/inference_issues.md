# Inference issues log

## Issue 1: Gemma 3 1B returns a garbage token in certain batches (SDPA attention)

**Found:** 2026-10-03, during the full baseline run (`task_1.ipynb` Step 2.6).
**Status:** fixed. Gemma 3 1B now uses `attn_implementation="eager"` and was rerun on all 1034 examples. The faulty run is kept locally (not committed) in `archive/gemma3-1b_baseline_sdpa_bug/`.

### Symptom
In the first full run, 6 of Gemma 3 1B's 1034 outputs were the same three tokens, `'ாரு'` + `<end_of_turn>` (the first one is token id 181327, a Tamil character sequence). These outputs had no SQL and were marked invalid: ids 846, 870, 872, 1004, 1014, 1015. The model produced no other unusual output. Gemma 3 4B and Gemma 4 E4B had no invalid outputs, and the 30-example 1B pilot had none either.

### Settings when it happened
| Setting | Value |
|---|---|
| Model | `google/gemma-3-1b-it`, loaded with `AutoModelForCausalLM` |
| Precision / attention | bf16, `attn_implementation="sdpa"` (PyTorch scaled-dot-product attention) |
| Batching | batch size 16, **left padding** (pad id 0); batches fixed from the dev ids sorted by prompt length |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=8192` |
| Input | the shared zero-shot prompt as one user turn, rendered with the Gemma 3 chat template |
| Software / hardware | torch 2.11.0+cu126, transformers 5.18.0, Python 3.12.13, 1× NVIDIA A40 46 GB |

The failures came from **2 of the 65 batches**:
| Batch | Padded width (tokens) | Input lengths | Failed in the full run |
|---|---|---|---|
| #1 | 417 | 406–417 | 872, 1015, 870, 1004, 1014 |
| #47 | 833 | 760–833 | 846 |

### Diagnosis
| Test | Result |
|---|---|
| Each failed example run alone (batch size 1, SDPA) | all 6 produce normal SQL |
| The same 2 batches rerun (SDPA, new process) | 15/16 broken in each batch (the full run had 5/16 and 1/16) |
| The same 2 batches, `attn_implementation="eager"` | 0 broken |
| SDPA forced to the **math** kernel (`sdpa_kernel(SDPBackend.MATH)`) | 0 broken |
| SDPA forced to the **memory-efficient** kernel (`SDPBackend.EFFICIENT_ATTENTION`) | 15/16 broken, the same as the default, so this is the kernel SDPA picked |
| Non-finite values (NaN/inf) in logits or hidden states, SDPA vs eager | **none** in either. The logits are finite but wrong: the top token is 181327, not the code fence that eager picks. |
| 3 other batches whose width is also ≡ 1 (mod 8) (widths 721, 1961, 2025) | 0 broken |
| Batch #1 repeated 5× in one process | 15/16 broken every time |

### Possible explanations (not confirmed)
- **A bug in the memory-efficient SDPA kernel for certain shapes plus an explicit attention mask.** This is the most likely explanation. The error follows that kernel exactly: the math kernel and eager attention are both correct on the same inputs. With left padding, transformers passes an explicit additive mask, and the memory-efficient kernel needs aligned memory for masks. Both failing batches have a width that is 1 more than a multiple of 8 (417, 833), which fits an alignment or mask-stride edge case. That pattern is not sufficient on its own, though: other widths ≡ 1 (mod 8) are fine, so some other feature of the shape or mask must matter too.
- **Run-to-run differences point to memory state.** The same batch broke 5/16 and 1/16 in the full run but 15/16 in later standalone runs, even though it repeats exactly within one process. That suggests the result depends on memory or kernel state, for example values left over from earlier work on the GPU. In the full run, 1B ran after E4B and 4B in the same process.
- **NaN from fully masked padding rows: ruled out.** No non-finite values appear anywhere, so this isn't the familiar "fully masked row → NaN" padding problem.
- **Not a model or prompt problem.** The same examples give normal SQL alone, with eager attention, or with the math kernel.

### Fix and why
- `ModelSpec.attn_implementation = "eager"` for Gemma 3 1B (`src/config.py`), and its full baseline run was redone from scratch so all 1034 outputs come from one setup.
- **Why not patch only the 6 examples:** that would mix two setups in one run. The bad batches also aren't reproducible, so patching would hide the problem rather than fix it.
- **Gemma 3 4B and Gemma 4 E4B stay on SDPA.** Neither produced an invalid or degenerate output (0/1034 each).
- **Cost:** eager is slower and uses more memory than SDPA, but 1B is small (~3 min per run).

### Result of the fix (2026-10-04)
- The garbage token appears **0 times** in the eager rerun. All 6 previously broken examples now give normal SQL (e.g. id 870 gives `SELECT name FROM Highschooler WHERE grade = 10`).
- 1 of the 1034 outputs is invalid (id 674): the model wrote `SELECTNationality, COUNT(*) ... GROUP BY SELECTNationality`, with no space after `SELECT`. This is a genuine model error, and the query would not run anyway, so it is correctly counted as invalid.
- **Reproducibility:** in a fresh process, eager reproduces the stored raw outputs byte for byte (58/58 examples in batches 1, 28, 47, 64, including both former bad batches).
- **Effect of switching attention:** the extracted SQL is identical to the old SDPA run for 864/1034 examples (83.6%). The rest differ because of normal near-tie effects in bf16, not because of the bug. This is consistent with the batching sensitivity described below.
- **Cost:** 0.15 s/example (SDPA: 0.14) and 6.1 GB peak GPU memory (SDPA: 4.7 GB).
- **Effect on the pilot:** the 30-example pilot (`task_1.ipynb` Step 2.5) was first run with SDPA for 1B. It was regenerated after the switch, so `archive/pilot/gemma3-1b/` now matches the current code and is what a rerun reproduces. Compared with the SDPA pilot, 27/30 1B outputs are identical; E4B and 4B are unchanged (30/30 identical, still SDPA). Pilot stats reported before 2026-10-04 (e.g. 1B at 0.16 s/example) come from the SDPA pilot, which is kept locally in `archive/gemma3-1b_pilot_sdpa/`. Neither pilot is committed: `archive/` is gitignored.

### Related observation: batching changes about 10% of outputs (separate from this bug)
On 96 random examples (6 batches), the extracted SQL from batched runs matched batch-size-1 runs in 84/96 (1B) and 87/96 (4B) with SDPA, and in 81/96 and 84/96 with eager. Rerunning the stored batches reproduced the stored outputs exactly. So outside the bug, the runs are deterministic. But in bf16, which examples share a padded batch can tip greedy decoding toward a different near-tied token. This is normal behavior for batched bf16 inference, but it is a reproducibility limitation to mention in the report: results are reproducible for a **fixed batch size and order** (`C.BATCH_SIZE = 16`, prompt-length order).
