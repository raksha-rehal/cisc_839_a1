# Evaluation metrics (Task 1, Step 3)

This file explains, in plain terms, how the generated SQL is scored: which metrics we use, where each comes from, how invalid outputs are counted, and a limitation of the official exact-match metric with how we handle it. The numbers come from `outputs/metrics/` (produced by `src/task_1.ipynb`, Step 3).

## Where the results are
All files are in `outputs/metrics/`, written by `src/task_1.ipynb` Step 3.4–3.6. They are deterministic, so a rerun gives identical files.

**Central results table:** **`task1_summary.csv`**, one row per model. It covers the assignment's "performance of all models, overall and by difficulty level, in a summary table" plus the efficiency measures, and is the main results table for the report:
- `EX_<level>` and `EM_<level>` for easy, medium, hard, extra and all;
- `EM_normalized_all`;
- `s_per_example_mean`;
- `output_tokens_mean`;
- `thinking_tokens_mean` (E4B only).

**Supporting detail:**

| File | Contents |
|---|---|
| `task1_accuracy_by_difficulty.csv` | EX, EM and EM (normalized) per model × difficulty level, with n (long format) |
| `task1_efficiency.csv` | Inference time per example (mean/median), total run time, output tokens (mean/median/max), thinking tokens (E4B) |
| `task1_efficiency_by_difficulty.csv` | The same time and token measures per difficulty level |
| `task1_component_f1.csv` | Precision/recall/F1 per SQL component (SELECT, WHERE, GROUP BY, ORDER BY, KEYWORDS) |
| `task1_invalid_outputs.csv` | No SQL / unparseable / still EX-correct / `INNER JOIN` / execution errors / truncated |
| `task1_model_agreement.csv` | Model vs model: accuracy difference and both-right / only-A / only-B / both-wrong counts, per metric and level |
| `task1_per_example.csv` | Every score and flag for every model × example. All other tables are built from this |
| `figures/*.png` | Accuracy by difficulty, accuracy vs. inference time, unparseable predictions |

## The metrics

| Metric | Question it answers | Source |
|---|---|---|
| **Exact Match (EM)** | Is the query *built* the same way as the gold query? | Spider paper |
| **Component F1** | How well does each *part* of the query match? | Spider paper |
| **Execution Accuracy (EX)** | Does the query return the *same answer* on the database? | Official Spider evaluation script / leaderboard (not reported in the paper) |
| **EM (normalized)** | EM after rewriting `INNER JOIN` → `JOIN` | **Our addition** (see below) |

### Exact Match (EM)
EM checks whether the predicted query is built the same way as the gold query. Both queries are split into parts: the selected columns, the tables, the conditions, grouping, ordering and keywords. EM = 1 only if **every** part matches.
- The order inside a part doesn't matter: `SELECT name, age` = `SELECT age, name`.
- **Values are ignored:** `WHERE age > 30` matches `WHERE age > 40`.

Example:
- `SELECT count(*) FROM singer` for gold `SELECT count(*) FROM singer` → **EM = 1**.
- `SELECT count(Singer_ID) FROM singer` → **EM = 0**, even though it returns the same number.

So EM is strict. It rewards writing the query the way the gold query is written.

### Component F1
Component F1 looks at one part at a time (SELECT, WHERE, GROUP BY, ORDER BY, KEYWORDS) and asks how much of that part is right:
- **precision:** how much of what the model wrote in that part is correct;
- **recall:** how much of the gold part the model got.

F1 combines the two. It shows *where* the queries go wrong; for example, a model may get SELECT right but WHERE wrong.

### Execution Accuracy (EX)
EX runs the predicted and the gold query on the real database and checks that they **return the same rows**.
- Row order only matters if the gold query has `ORDER BY`.
- The `count(Singer_ID)` example above gets **EX = 1**.

EX is closer to "did the model answer the question?", and it isn't affected by writing style. It can occasionally count a wrong query as right if the query happens to return the same rows on this particular database. The Spider authors note this, and it's why they left EX for future work in the paper.

### Settings (official script defaults)
- **Code:** `taoyds/test-suite-sql-eval`, pinned at commit `e97acc5`.
- **EX database:** the original Spider dev database for each question (`data/spider/database/<db_id>/<db_id>.sqlite`). The distilled "test-suite" databases are not used.
- **EX values:** the model's own predicted values (`plug_value=False`).
- **DISTINCT:** removed before comparing results (`keep_distinct=False`).
- **Timeout:** 60 s per query.
- **EM:** ignores values, as in the paper.
- **Difficulty levels** (easy / medium / hard / extra) come from the same script.
- **Cross-check:** our per-example scores reproduce the script's printed EX, EM and component F1 exactly for every model and difficulty level (`task_1.ipynb` Step 3.3).
- **A quirk we reproduce:** when a component's precision and recall are both 0, the official script sets its F1 to 1.

## Efficiency measures
Efficiency is not part of the Spider paper; the assignment asks for it. It is reported overall (`task1_efficiency.csv`, `task1_summary.csv`) and **by difficulty level** (`task1_efficiency_by_difficulty.csv`). We added the per-level breakdown because it shows whether the reasoning model spends more effort on harder questions.

- **Output tokens per example:** generated tokens up to the end-of-turn token. For E4B this includes its reasoning; **thinking tokens** counts just the reasoning part. These are exact per-example counts, so they are the most reliable efficiency measure by difficulty.
- **Inference time per example:** the wall-clock time of a batch of 16 divided by 16, on one NVIDIA A40 in bf16. Batches are grouped by prompt length, not by difficulty, so the per-difficulty times are partly averaged across levels. Use them as a rough guide, and use the token counts to compare effort across levels.

E4B's thinking grows steadily with difficulty: 138.5 tokens on easy questions, 223.9 on medium, 405.1 on hard and 539.2 on extra. The Gemma 3 models' outputs also get longer on harder questions, but only from about 19–20 tokens to 30–65.

## How invalid outputs are counted
Every output that is wrong for a technical reason counts as **incorrect (0) on every metric**, and is counted separately in `outputs/metrics/task1_invalid_outputs.csv`:
- **no SQL extracted** from the model's answer;
- **unparseable:** the official parser can't read the SQL (see the next section);
- **execution error:** the SQL fails when run, e.g. a column that doesn't exist;
- **truncated:** the output was cut off at the 8192-token cap. This happened 0 times for all models.

## A limitation of EM: the official parser rejects some valid SQL

### Why the parser fails
EM doesn't run the query. It takes the predicted SQL apart into pieces and compares them with the gold query's pieces. To do that, it first has to **read** the SQL, using a small hand-written parser (`process_sql.py`) that the Spider authors wrote in 2018.

That parser only understands the style the **gold** queries are written in. The gold queries always write `JOIN` and never `INNER JOIN`, so the parser was never taught `INNER JOIN`, and it gives up when it sees one. It's like a spell-checker trained on British English flagging "color". It also doesn't understand some other valid SQL:
- value lists such as `IN ('2014', '2015')`;
- table aliases without `AS` (`FROM singer s`);
- column aliases (`COUNT(*) AS cnt`);
- double-quoted column names;
- `CAST(...)`.

When the parser gives up, the prediction is treated as **empty**, so EM = 0, however correct the SQL was. The same happens to component F1. EX is not affected, because it runs the query instead of reading it.

**Why Gemma 4 E4B is hit hardest:** it often writes `INNER JOIN` (342 of its 1034 queries). The Gemma 3 models mostly write a bare `JOIN`, like the gold queries.

| Model | Unparseable | …of which still EX-correct | Use `INNER JOIN` | Parse after `INNER JOIN` → `JOIN` |
|---|---|---|---|---|
| gemma-3-1b-it | 473 | 64 (14%) | 5 | 0 |
| gemma-3-4b-it | 290 | 148 (51%) | 75 | 60 |
| gemma-4-E4B-it | 422 | 306 (73%) | 342 | 306 |

For Gemma 3 1B, most unparseable predictions are genuine mistakes (only 14% return the right answer).

### What "EM (normalized)" means
`INNER JOIN` and `JOIN` are **exactly the same thing** in SQL: `JOIN` is just shorthand for `INNER JOIN`. "EM (normalized)" applies one find-and-replace to the predicted SQL before scoring, `INNER JOIN` → `JOIN`, and then computes the official EM exactly as before. Nothing else is changed. It's like converting "color" to "colour" before running the British spell-checker: the meaning doesn't change, it only stops the checker from rejecting the word.

- **Our addition:** this column is **not** part of the Spider paper or the official script. **Official EM stays the main EM number.**
- **Why only this rewrite:** it's the only one that is always exactly equivalent. The other unsupported styles have edge cases; for example, double quotes can mean a text value in SQLite.

### Effect on the results

| Model | EM (official) | EM (normalized) | EX |
|---|---|---|---|
| gemma-3-1b-it | 0.183 | 0.183 (no change) | 0.300 |
| gemma-3-4b-it | 0.396 | 0.410 (+1.4 pts) | 0.707 |
| gemma-4-E4B-it | 0.388 | **0.502 (+11.4 pts)** | 0.838 |

For E4B, EM by difficulty rises from 0.404 to 0.576 (medium), 0.247 to 0.368 (hard), and 0.018 to 0.114 (extra).

- **With official EM only,** E4B looks slightly *worse* than Gemma 3 4B (0.388 vs 0.396) while being much better on EX (0.838 vs 0.707). A reader could wrongly conclude that reasoning doesn't help with getting the query structure right.
- **With the normalized column,** E4B leads on both metrics, and the gap between the two EM columns shows how much of the official EM gap comes from the parser.
- **Not changed by this:** EX and the official EM numbers.
- **Component F1 has the same issue:** it uses the same parser, so it is also pulled down for E4B.

## References
- Yu et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP. arXiv:1809.08887. The "Evaluation Metrics" section defines Component Matching and Exact Matching. Values are not evaluated. Execution Accuracy is discussed but not reported, because it needs value prediction and can give false positives.
- Official evaluation code: https://github.com/taoyds/test-suite-sql-eval (commit `e97acc5`), which computes EX, EM, component scores and difficulty levels.
