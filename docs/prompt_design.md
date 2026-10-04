# Prompt design (Task 1, zero-shot)

The template text is in `src/prompt_template.txt`. All three models get the same prompt for every dev example.

## Template
```
You are an expert in SQLite. Given the database schema and the question below, write one SQLite query that answers the question.

### Database: {db_id}
{schema}

### Question
{question}

### Instructions
- Use only the tables and columns defined in the schema above.
- Write exactly one SQL query that is valid in SQLite.
- Respond with ONLY the SQL query. Do not include any explanation, comments, or Markdown code fences.
```
- `{db_id}`, `{question}`: taken directly from the `dev.json` entry.
- `{schema}`: CREATE TABLE DDL + 3 sample rows per table (described below).

## Schema representation: CREATE TABLE + 3 sample rows
- The schema isn't stored in `dev.json`; it comes from `tables.json`, joined on `db_id`.
- Table and column names are the **original** ones (`table_names_original`, `column_names_original`), because the gold SQL uses them.
- Column types are uppercased (`TEXT`, `NUMBER`, `TIME`), and `others` is mapped to `TEXT`.
- PK columns are marked `PRIMARY KEY`. Each FK gets a `FOREIGN KEY (col) REFERENCES table(col)` line.
- After each CREATE TABLE there's a comment block in the Rajkumar et al. (2022) format, filled with `SELECT * FROM <table> LIMIT 3` from `database/<db_id>/<db_id>.sqlite`. Values are tab-separated, and cells longer than 100 chars are truncated.
- The DDL is built from `tables.json`, not copied from each DB's `sqlite_master`. That keeps the format identical across all 20 dev DBs, since the raw DDL varies in quoting, comments and layout.

Example (`concert_singer`; the rows are illustrative):
```
CREATE TABLE concert (
  concert_ID NUMBER PRIMARY KEY,
  concert_Name TEXT,
  Theme TEXT,
  Stadium_ID TEXT,
  Year TEXT,
  FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
);
/*
3 example rows:
SELECT * FROM concert LIMIT 3;
concert_ID	concert_Name	Theme	Stadium_ID	Year
...
*/
```

Size: dev DBs have 2–11 tables and 7–56 columns. The largest DDL (`student_transcripts_tracking`) is about 2.5k chars before the sample rows are added.

## Why DDL + 3 rows instead of DDL only
1. **Value grounding.** A schema only gives names and types; it doesn't show how values are written (e.g. `Is_male` stores `'T'/'F'`, or `'F'` vs `'female'`, code vs full name, date formats). A wrong literal returns an empty or wrong result, so **Execution Accuracy** fails even when the query structure is right.
2. **Evidence from the literature** (each claim checked against the paper's full text on 2026-10-03):
   - Rajkumar, Li & Bahdanau (2022). *Evaluating the Text-to-SQL Capabilities of Large Language Models*. arXiv:2204.00498. In their zero-shot comparison of prompt formats on Spider dev (Table 2), "Create Table + Select 3" scored highest, at 67.0% execution accuracy (Codex davinci). We use their format.
   - Chang & Fosler-Lussier (2023). *How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings*. arXiv:2305.11853. They write that "database content is valuable, but its representation needs to be carefully chosen". For zero-shot they recommend "normalized CreateTable SelectCol". **Caveat:** the column-wise `SelectCol 3` format beat row-wise formats in their experiments (by 3.8 points over InsertRow for Codex). That makes switching to SelectCol a possible Task 3 prompting improvement.
   - Gao et al. (2023). *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation* (DAIL-SQL). arXiv:2308.15363. In their ablations, foreign keys improved execution accuracy by 0.6–2.9% for OpenAI models (mixed for some representations), so we include FK lines. A "with no explanation" rule "consistently boosts" EM (by more than 6%) and EX (by more than 3%), which supports the "respond with ONLY the SQL query" instruction. They did **not** find CREATE TABLE to be the best overall format, so we cite them only for FKs and the no-explanation rule.
3. **Cost.** Every prompt fits easily in context. The extra tokens only lengthen prefill, and output-token counts aren't affected.
4. **Fairness.** Every model and example gets the same fixed, deterministic context. There's no per-question retrieval, and no gold SQL can leak into the prompt.

**Limitation:** 3 rows may not include the value a particular question needs. They're a sample, not full value retrieval.

## How the prompt is sent to each model
- The full template goes in a **single user turn** for all models. Gemma 3 has no real system role (its chat template folds system text into the first user turn), so this keeps the inputs identical.
- Gemma 4 E4B: thinking is enabled with `enable_thinking=True`, which makes the chat template put `<|think|>` at the start of the system turn. In a probe, E4B still **skipped thinking on most easy questions** and answered with bare SQL, which would leave those examples without a trace. So the model turn is also pre-filled with the thought-channel opener `<|channel>thought\n` (`config.THINKING_PREFILL`), which forces a reasoning trace for every example. This follows DeepSeek-R1's documented advice to force the `<think>\n` opener. The user message is still identical across models; only the start of E4B's own turn differs. The trace is saved in full and kept separate from the final answer.
- SQL extraction: strip Markdown fences, take the first SQL statement, and collapse it to one line (the Spider eval script reads one prediction per line). Any output that can't be parsed is counted as invalid, which means incorrect, and the number of invalid outputs is reported per model.
