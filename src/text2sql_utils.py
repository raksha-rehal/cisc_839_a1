"""Shared code for task_1.ipynb and task_3.ipynb.

Task 1 and Task 3 must build prompts, generate, extract SQL and score in exactly the same way, so the code
lives here once. The notebooks explain each step in their markdown cells and call these functions.

Sections: Spider data · schema + prompt · official evaluation script · model + generation · scoring.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config as C

# ---------------------------------------------------------------- Spider data


def db_path(db_id: str) -> Path:
    """SQLite file of a Spider database."""
    return C.SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"


def load_spider(split: str) -> list[dict]:
    """A Spider split as downloaded: 'dev' -> dev.json, 'train_spider' -> train_spider.json."""
    return json.loads((C.SPIDER_DIR / f"{split}.json").read_text())


def load_tables() -> dict[str, dict]:
    """Schema of every Spider database (tables.json), keyed by db_id."""
    return {t["db_id"]: t for t in json.loads((C.SPIDER_DIR / "tables.json").read_text())}


# ---------------------------------------------------------------- schema + prompt
TYPE_MAP = {"others": "text"}  # tables.json type -> type shown in the DDL


def ident(name: str) -> str:
    """Name as written in the DDL: quoted only if it is not a plain identifier."""
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'


def fmt_cell(value) -> str:
    s = "NULL" if value is None else re.sub(r"[\t\r\n]+", " ", str(value))
    return s if len(s) <= C.MAX_CELL_CHARS else s[: C.MAX_CELL_CHARS] + "..."


def sample_rows(conn, table: str, columns: list[str], fallbacks: list | None = None) -> tuple[list[str], list[tuple]]:
    quoted = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
    try:
        cur = conn.execute(f'SELECT {quoted} FROM "{table}" LIMIT {C.N_SAMPLE_ROWS}')
        return columns, cur.fetchall()
    except sqlite3.OperationalError:
        if fallbacks is not None:
            fallbacks.append(table)
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT {C.N_SAMPLE_ROWS}')
        return [d[0] for d in cur.description], cur.fetchall()


def render_schema(t: dict, db_file: Path, fallbacks: list | None = None) -> str:
    """CREATE TABLE statements + 3 sample rows per table (the schema part of the prompt)."""
    names, cols, types = t["table_names_original"], t["column_names_original"], t["column_types"]
    pks = {i for pk in t["primary_keys"] for i in (pk if isinstance(pk, list) else [pk])}
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")  # a few DBs have bad UTF-8
    blocks = []
    for ti, table in enumerate(names):
        idx = [i for i, (tj, _) in enumerate(cols) if tj == ti]
        lines = [
            f"  {ident(cols[i][1])} {TYPE_MAP.get(types[i], types[i]).upper()}" + (" PRIMARY KEY" if i in pks else "")
            for i in idx
        ]
        lines += [
            f"  FOREIGN KEY ({ident(cols[a][1])}) REFERENCES {ident(names[cols[b][0]])}({ident(cols[b][1])})"
            for a, b in t["foreign_keys"]
            if cols[a][0] == ti
        ]
        header, rows = sample_rows(conn, table, [cols[i][1] for i in idx], fallbacks)
        sample = [
            "/*",
            f"{C.N_SAMPLE_ROWS} example rows:",
            f"SELECT * FROM {ident(table)} LIMIT {C.N_SAMPLE_ROWS};",
            "\t".join(header),
            *("\t".join(fmt_cell(v) for v in row) for row in rows),
            "*/",
        ]
        blocks.append(f"CREATE TABLE {ident(table)} (\n" + ",\n".join(lines) + "\n);\n" + "\n".join(sample))
    conn.close()
    return "\n\n".join(blocks)


def build_prompt(template: str, db_id: str, schema: str, question: str) -> str:
    """Fill the plain-text prompt template (src/prompt_template.txt)."""
    return template.format(db_id=db_id, schema=schema, question=question)


# ---------------------------------------------------------------- official evaluation script


def setup_eval_script() -> None:
    """Clone the official Spider evaluation code at the pinned commit and make it importable."""
    import nltk

    if not (C.EVAL_SCRIPT_DIR / "evaluation.py").exists():
        subprocess.run(["git", "clone", "--quiet", C.EVAL_SCRIPT_REPO, str(C.EVAL_SCRIPT_DIR)], check=True)
    subprocess.run(["git", "-C", str(C.EVAL_SCRIPT_DIR), "checkout", "--quiet", C.EVAL_SCRIPT_COMMIT], check=True)
    nltk.download("punkt_tab", quiet=True)
    if str(C.EVAL_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(C.EVAL_SCRIPT_DIR))


def _ensure_eval_importable() -> None:
    if str(C.EVAL_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(C.EVAL_SCRIPT_DIR))


@functools.cache
def eval_schema(db_id: str):
    """The official parser's view of a database schema (cached per database)."""
    _ensure_eval_importable()
    from process_sql import Schema, get_schema

    return Schema(get_schema(str(db_path(db_id))))


@functools.cache
def _foreign_key_maps():
    _ensure_eval_importable()
    from evaluation import build_foreign_key_map_from_json

    return build_foreign_key_map_from_json(str(C.SPIDER_DIR / "tables.json"))


def difficulty(db_id: str, gold_query: str) -> str:
    """Official Spider difficulty (easy / medium / hard / extra) of a gold query."""
    _ensure_eval_importable()
    from evaluation import Evaluator
    from process_sql import get_sql

    return Evaluator().eval_hardness(get_sql(eval_schema(db_id), gold_query))


# ---------------------------------------------------------------- model + generation


def load_model(spec, model_path: str | Path | None = None):
    """Load a model and its tokenizer/processor.

    model_path=None loads the pinned Hugging Face revision of spec.hf_id; a local path loads a fine-tuned checkpoint
    (Task 3) with the same class and attention implementation.
    """
    import torch
    import transformers

    token = C.get_hf_token() if spec.gated and model_path is None else None
    source = str(model_path) if model_path is not None else spec.hf_id
    revision = None if model_path is not None else spec.revision
    proc_cls = transformers.AutoTokenizer if spec.loader == "AutoModelForCausalLM" else transformers.AutoProcessor
    processor = proc_cls.from_pretrained(source, revision=revision, token=token)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"  # decoder-only batching
    model = getattr(transformers, spec.loader).from_pretrained(
        source, revision=revision, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=spec.attn_implementation, token=token,
    ).eval()
    return model, processor, tokenizer


def build_input(spec, processor, prompt: str) -> str:
    """The prompt as one user turn in the model's own chat template (+ the thinking opener for E4B)."""
    extra = {"enable_thinking": True} if spec.reasoning else {}
    text = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **extra
    )
    return text + (C.THINKING_PREFILL if spec.reasoning else "")


FENCE = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)(?:```|$)", re.S | re.I)
SQL_START = re.compile(r"\bSELECT\b|\bWITH\s+\w+\s+AS\s*\(", re.I)


def extract_sql(answer: str) -> str:
    """One SQL query from a model answer (same rules for every model; see task_1 Step 2.3)."""
    text = answer or ""
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = SQL_START.search(text)
    if not start:
        return ""
    text, quote = text[start.start() :], None
    for i, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"`":
            quote = ch
        elif ch == ";" or text.startswith("\n\n", i):
            text = text[:i]
            break
    return " ".join(text.split())


def split_output(spec, processor, gen_ids: list[int]) -> tuple[str, str | None, str]:
    """Return (raw_output, thinking, answer) for one example's generated token ids."""
    if not spec.reasoning:
        raw = processor.decode(gen_ids, skip_special_tokens=False)
        return raw, None, processor.decode(gen_ids, skip_special_tokens=True).strip()
    raw = C.THINKING_PREFILL + processor.decode(gen_ids, skip_special_tokens=False)
    if "<channel|>" not in raw:  # cap reached while still thinking
        trace = processor.decode(gen_ids, skip_special_tokens=True)
        return raw, trace.strip(), ""
    parsed = processor.parse_response(raw, prefix="")
    return raw, (parsed.get("thinking") or "").strip(), (parsed.get("content") or "").strip()


def read_records(key: str, run_name: str) -> dict[int, dict]:
    """Per-example outputs of a run, keyed by example id."""
    path = C.run_path(key, run_name) / "responses.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def run_model(key: str, ids: list[int], run_name: str, prompts, model_path=None, **extra_config) -> None:
    """Generate outputs for the given example ids (resumable; one JSON line per example).

    prompts: prompt text per example id (a list or dict indexable by id).
    model_path: optional local checkpoint (Task 3); otherwise the pinned Hugging Face model.
    """
    import torch
    from tqdm.auto import tqdm

    spec, dec = C.MODELS[key], C.DECODING[key]
    out_path = C.run_dir(key, run_name) / "responses.jsonl"
    done = read_records(key, run_name)
    # Batches are fixed from the full id list, so batch composition (and hence the bf16
    # numerics) is the same whether or not the run was interrupted and resumed.
    ordered = sorted(ids, key=lambda i: (len(prompts[i]), i))
    batches = [ordered[b : b + C.BATCH_SIZE] for b in range(0, len(ordered), C.BATCH_SIZE)]
    batches = [batch for batch in batches if any(i not in done for i in batch)]
    n_todo = sum(i not in done for i in ids)
    print(f"{spec.name} [{run_name}]: {len(ids) - n_todo}/{len(ids)} done, {n_todo} to go")
    if not batches:
        return
    if model_path is not None:
        extra_config["model_path"] = str(model_path)
    C.save_run_config(
        key, run_name, batch_size=C.BATCH_SIZE, n_examples=len(ids),
        prompt_template=str(C.PROMPT_TEMPLATE.relative_to(C.REPO_ROOT)),
        thinking_prefill=C.THINKING_PREFILL if spec.reasoning else None, **extra_config,
    )
    C.set_seed()
    model, processor, tokenizer = load_model(spec, model_path)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    torch.cuda.reset_peak_memory_stats()

    with out_path.open("a", encoding="utf-8") as f:
        for batch in tqdm(batches, desc=spec.name):
            texts = [build_input(spec, processor, prompts[i]) for i in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=dec.max_new_tokens, do_sample=dec.do_sample,
                    temperature=dec.temperature, top_p=dec.top_p, top_k=dec.top_k,
                )
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            width = enc["input_ids"].shape[1]
            for j, i in enumerate(batch):
                if i in done:  # regenerated only to keep the batch identical
                    continue
                gen = [t for t in out[j, width:].tolist() if t != tokenizer.pad_token_id]
                raw, thinking, answer = split_output(spec, processor, gen)
                query = extract_sql(answer)
                rec = {
                    "id": i,
                    "query": query,
                    "answer": answer,
                    "raw_output": raw,
                    "thinking": thinking,
                    "valid": bool(query),
                    "truncated": not gen or gen[-1] not in eos_ids,
                    "input_tokens": int(enc["attention_mask"][j].sum()),
                    "output_tokens": len(gen),
                    "thinking_tokens": len(tokenizer(thinking, add_special_tokens=False)["input_ids"]) if thinking else 0,
                    "latency_s": wall / len(batch),
                    "batch_wall_s": wall,
                    "batch_size": len(batch),
                    "peak_mem_gb": round(peak, 2),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    del model, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def summarize(key: str, run_name: str, n_full: int) -> dict:
    """Speed / length / validity summary of a run (n_full: size of the full run, for the time projection)."""
    import statistics

    recs = list(read_records(key, run_name).values())
    total = sum(r["latency_s"] for r in recs)
    return {
        "model": C.MODELS[key].name,
        "n": len(recs),
        "s/example": round(total / len(recs), 2),
        "mean output tokens": round(statistics.mean(r["output_tokens"] for r in recs), 1),
        "max output tokens": max(r["output_tokens"] for r in recs),
        "mean thinking tokens": round(statistics.mean(r["thinking_tokens"] for r in recs), 1),
        "truncated": sum(r["truncated"] for r in recs),
        "invalid": sum(not r["valid"] for r in recs),
        "peak GPU mem (GB)": max(r["peak_mem_gb"] for r in recs),
        "projected full run (h)": round(total / len(recs) * n_full / 3600, 2),
    }


# ---------------------------------------------------------------- scoring (official Spider evaluation)
# what evaluate() scores a prediction as when the parser can't read it
EMPTY_SQL = {"except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [], "intersect": None,
             "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []}
COMPONENTS = {"select": "SELECT", "where": "WHERE", "group": "GROUP BY", "order": "ORDER BY", "keywords": "KEYWORDS"}
INNER_JOIN = re.compile(r"\bINNER\s+JOIN\b", re.I)


def parse_pred(schema, p_str):
    _ensure_eval_importable()
    from process_sql import get_sql

    if not p_str:
        return EMPTY_SQL, False
    try:
        return get_sql(schema, p_str), True
    except Exception:
        return EMPTY_SQL, False


def exact_match(db_id, gold_str, p_sql):
    """Official EM + component scores (same steps as evaluate() in evaluation.py)."""
    _ensure_eval_importable()
    from evaluation import Evaluator, build_valid_col_units, rebuild_sql_col, rebuild_sql_val
    from process_sql import get_sql

    schema, kmap = eval_schema(db_id), _foreign_key_maps()[db_id]
    g_sql = get_sql(schema, gold_str)
    g_sql = rebuild_sql_col(build_valid_col_units(g_sql["from"]["table_units"], schema), rebuild_sql_val(g_sql), kmap)
    p_sql = rebuild_sql_col(build_valid_col_units(p_sql["from"]["table_units"], schema), rebuild_sql_val(p_sql), kmap)
    ev = Evaluator()
    return int(ev.eval_exact_match(p_sql, g_sql)), ev.partial_scores


def score_example(ex: dict, pred: str) -> dict:
    """EX, EM, EM (normalized), component scores and failure flags for one prediction (see task_1 Step 3.2)."""
    _ensure_eval_importable()
    from exec_eval import eval_exec_match, exec_on_db, postprocess, remove_distinct

    db_id, schema, db = ex["db_id"], eval_schema(ex["db_id"]), str(db_path(ex["db_id"]))
    p_str = pred.replace("value", "1")  # same preprocessing as evaluate()
    p_sql, parsed = parse_pred(schema, p_str)
    em, partial = exact_match(db_id, ex["query"], p_sql)
    p_sql_norm, parsed_norm = parse_pred(schema, INNER_JOIN.sub("JOIN", p_str))
    em_norm, _ = exact_match(db_id, ex["query"], p_sql_norm)
    if p_str:
        ex_score = eval_exec_match(db=db, p_str=p_str, g_str=ex["query"], plug_value=False, keep_distinct=False,
                                   progress_bar_for_each_datapoint=False)
        exec_error = asyncio.run(exec_on_db(db, remove_distinct(postprocess(p_str))))[0] == "exception"
    else:  # nothing to run: wrong (an empty query would return no rows and could "match" an empty gold result)
        ex_score, exec_error = 0, False
    row = {"em": em, "em_normalized": em_norm, "ex": int(ex_score), "parsed": parsed, "parsed_normalized": parsed_norm,
           "exec_error": exec_error, "uses_inner_join": bool(INNER_JOIN.search(p_str))}
    for c in COMPONENTS:
        s = partial[c]
        row |= {f"{c}_acc": s["acc"], f"{c}_rec": s["rec"], f"{c}_has_pred": s["pred_total"] > 0, f"{c}_has_gold": s["label_total"] > 0}
    return row


def score_run(examples: dict, predictions: dict, ids: list[int], workers: int = 8) -> list[dict]:
    """score_example for every id, in worker threads (the official execution check calls asyncio.run,
    which cannot run inside Jupyter's own event loop)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda i: score_example(examples[i], predictions[i]), ids))


ROW = re.compile(r"^(count|execution|exact match|select|where|group|order|keywords)\s+(.*)$")


def official_scores(examples: dict, predictions: dict, ids: list[int]) -> dict:
    """Run the official evaluation.py on gold/prediction files; return metric -> [easy, medium, hard, extra, all]."""
    with tempfile.TemporaryDirectory() as tmp:
        gold, pred = Path(tmp) / "gold.txt", Path(tmp) / "pred.txt"
        gold.write_text("".join(f"{examples[i]['query'].strip()}\t{examples[i]['db_id']}\n" for i in ids))
        pred.write_text("".join((predictions[i] or "SELECT") + "\n" for i in ids))
        out = subprocess.run(
            [sys.executable, "evaluation.py", "--gold", str(gold), "--pred", str(pred), "--etype", "all",
             "--db", str(C.SPIDER_DIR / "database"), "--table", str(C.SPIDER_DIR / "tables.json")],
            cwd=C.EVAL_SCRIPT_DIR, capture_output=True, text=True, check=True).stdout
    found, section = {}, None
    for line in out.splitlines():
        if "PARTIAL MATCHING" in line:
            section = line
        m = ROW.match(line)
        if m and (m.group(1) in ("count", "execution", "exact match") or "F1" in (section or "")):
            found[m.group(1)] = [float(v) for v in m.group(2).split()]
    return found


def component_scores(df, c):
    """Official aggregation: precision over examples with a predicted part, recall over examples with a gold part."""
    acc = df.loc[df[f"{c}_has_pred"], f"{c}_acc"].mean() if df[f"{c}_has_pred"].any() else 0.0
    rec = df.loc[df[f"{c}_has_gold"], f"{c}_rec"].mean() if df[f"{c}_has_gold"].any() else 0.0
    f1 = 1.0 if acc == 0 and rec == 0 else 2 * acc * rec / (acc + rec)  # the official script uses 1.0 when both are 0
    return acc, rec, f1


def level_slice(df, level):
    return df if level == "all" else df[df["difficulty"] == level]
