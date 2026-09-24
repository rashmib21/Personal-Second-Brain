"""
Domain-Agnostic Structured Spreadsheet Query Engine with LLM Query Planning.

This module provides a generic, data-driven query engine for structured
workbooks (.xlsx, .xls, .csv, .ods).

Architecture Flow:
1. User Query + Conversational History
2. Runtime Schema Grounding (extract sheets, columns, types, sample values, row counts)
3. Structured Query Plan Generation (LLM converts NL to machine-readable JSON plan)
4. Schema & Value Validation (validate fields/values against actual runtime data, fail closed if ungrounded)
5. Deterministic Pandas/LanceDB Execution (execute filtering, sorting, group-by, aggregation, distinct count)
6. Result Completeness Verification (format complete results without silent line limits)

Absolute Principles:
- NEVER hardcode business or domain concepts (e.g. companies, CTC, salary, products, students, cities, roles).
- Execution layer is strictly deterministic.
- Truncation to 10 rows is strictly forbidden unless explicitly requested by user (e.g. "top 5").
"""

import csv
import os
import re
import json
import pandas as pd
from openpyxl import load_workbook
from app.storage.lancedb_store import get_spreadsheet_table
from app.llm.ollama_client import ask_llama, ask_spreadsheet_llm
from app.services.interaction_state import (
    get_last_interaction,
    set_spreadsheet_context,
    get_spreadsheet_context
)



def normalize_string(value):
    """Normalize text by stripping whitespace and converting to lowercase."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()

# generic English shorthand only (not domain concepts); extend if your sheets use others
_ABBREV = {
    "dev": "developer", "devs": "developer", "eng": "engineer", "engg": "engineer",
    "mgr": "manager", "sr": "senior", "jr": "junior", "admin": "administrator",
    "ops": "operations", "exp": "experience", "asst": "assistant",
}


def _words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _canon(word):
    return _STEMMER.stem(_ABBREV.get(word, word))


def token_mask(series, token):
    """Rows of `series` containing `token` (stem-aware: developers == developer == 'Dev')."""
    t = _canon(token)
    mask = series.fillna("").astype(str).map(lambda x: t in {_canon(w) for w in _words(x)})
    return mask, "exact"


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that describe HOW the user asks, never WHAT they filter on. One list, used everywhere.
REQUEST_WORDS = {
    "provide", "give", "show", "list", "tell", "find", "get", "fetch", "display", "share", "want", "need",
    "please", "kindly", "hey", "hi", "hello", "claude", "could", "would", "should", "let", "know", "see",
    "name", "names", "record", "records", "row", "rows",
    "entry", "entries", "item", "items", "file", "files", "sheet", "sheets", "spreadsheet", "workbook",
    "table", "column", "columns", "listed", "mentioned", "available", "hiring", "hire", "hired", "hires",
    "recruiting", "looking", "offer", "offering", "offers", "having", "opening", "openings", "vacancy",
    "vacancies", "around", "based", "located", "situated", "working", "related", "regarding",
}
OPERATION_WORDS = {
    "highest", "lowest", "top", "bottom", "rank", "ranked", "sort", "sorted", "order", "ascending",
    "descending", "average", "avg", "mean", "sum", "total", "max", "maximum", "min", "minimum", "count",
    "number", "best", "worst", "largest", "smallest", "cheapest", "least", "most", "per", "each", "group",
    "grouped", "greater", "above", "below", "higher", "lower", "atleast", "atmost", "between",
}
# Words that can be real content ("Data Engineer") or pure filler ("show me the data"):
# they take part in phrase matching, but are silently dropped if they cannot ground on their own.
SOFT_WORDS = {"data", "info", "information", "detail", "details"}
FILE_WORDS = {"xlsx", "xls", "csv", "ods", "pdf", "docx", "txt", "md"}
FILLER = set(ENGLISH_STOP_WORDS) | REQUEST_WORDS | OPERATION_WORDS | FILE_WORDS


def strip_file_mentions(q_norm, file_path=None):
    """Remove file names the user typed (so 'IT_Direct_Hire_Companies_2026.xlsx' never becomes a filter)."""
    q = re.sub(r"\S+\.(?:xlsx|xls|csv|ods)\b", " ", q_norm)
    if file_path:
        stem = os.path.splitext(os.path.basename(file_path))[0].lower()
        q = q.replace(stem, " ")
    return q


def phrase_mask(series, tokens):
    """Rows where `tokens` occur as a contiguous phrase inside ONE list item of a cell ('Data Engineer' != 'Data Analyst, ML Engineer')."""
    tgt = [_canon(t) for t in tokens]

    def hit(cell):
        for seg in re.split(r"[,;|\n]", str(cell).lower()):
            ws = [_canon(w) for w in _words(seg)]
            for i in range(len(ws) - len(tgt) + 1):
                if ws[i:i + len(tgt)] == tgt:
                    return True
        return False

    return series.fillna("").astype(str).map(hit)


def classify_words(text, headers, sheet_names):
    """[(word, kind)] with kind in skip | header | sheet | content."""
    h_stems = {_canon(w) for h in headers for w in _words(h)}
    s_stems = {_canon(w) for s in sheet_names for w in _words(s)}
    out = []
    for w in _words(text):
        if w in SOFT_WORDS:
            out.append((w, "soft"))
        elif len(w) <= 2 or w in FILLER or re.fullmatch(r"\d+(\.\d+)?", w):
            out.append((w, "skip"))
        elif _canon(w) in h_stems:
            out.append((w, "header"))
        elif _canon(w) in s_stems:
            out.append((w, "sheet"))
        else:
            out.append((w, "content"))
    return out


def content_runs(classified):
    """Consecutive content/soft words form one candidate phrase ('data engineer'). Runs with no hard content word are dropped."""
    runs, cur = [], []
    for w, kind in classified:
        if kind in ("content", "soft"):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    hard = {w for w, k in classified if k == "content"}
    return [r for r in runs if any(w in hard for w in r)]


def parse_numeric_value(value):
    """
    Parses a cell value into float or None.
    Handles numeric formats:
      - Integers / Floats: 80000, 50.5
      - Currency / Range strings: $500, €40, ₹100, 10-20 (returns max 20.0 for range ranking)
    Excludes identifier strings like 'T101', 'INV-2026'.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # Exclude alphanumeric IDs like T101, INV_001
    if re.search(r"^[a-zA-Z]{1,5}[-_]?\d+$", text):
        return None

    cleaned_text = text.replace(",", "").replace("$", "").replace("₹", "").replace("€", "")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned_text)

    if numbers:
        floats = [float(num) for num in numbers]
        return max(floats)

    return None


def find_header_row_index(values):
    """
    Generic header row detection for sheets with title/banner rows.
    Header rows have multiple non-empty distinct string cells across columns.
    """
    best_idx = 0
    best_score = -1.0

    for idx, row in enumerate(values[:25]):
        if not row:
            continue

        cells = [str(v).strip() for v in row if v is not None and str(v).strip()]
        if len(cells) < 2:
            continue

        non_numeric = [c for c in cells if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", c)]
        unique_set = set(c.lower() for c in non_numeric)

        # Header rows have multiple distinct text columns
        score = len(unique_set) * 3 + len(cells)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def is_valid_data_row(row_dict, headers):
    """
    Generic validation for a data row.
    Excludes empty rows, section headers (e.g. '── SECTION ──'), and single-cell legend rows.
    """
    non_empty_vals = [str(v).strip() for v in row_dict.values() if v is not None and str(v).strip()]
    if not non_empty_vals:
        return False

    # Section divider / title row (only 1 non-empty cell across many columns)
    if len(headers) >= 3 and len(non_empty_vals) == 1:
        first_val = non_empty_vals[0]
        if first_val.startswith("──") or first_val.endswith("──") or "🟣" in first_val or "🟢" in first_val:
            return False
        if len(first_val) > 40:
            return False

    return True


def inspect_sheet_schema(df, sheet_name=""):
    """
    Profiles a DataFrame dynamically to determine:
    1. Clean header names
    2. Column data types ('numeric' vs 'text')
    3. Primary entity/identifier column based on statistical uniqueness distribution
    """
    headers = [str(col).strip() for col in df.columns]

    column_types = {}
    for col in headers:
        series = df[col].dropna()
        if len(series) == 0:
            column_types[col] = "text"
            continue

        numeric_count = 0
        for val in series:
            parsed = parse_numeric_value(val)
            if parsed is not None:
                numeric_count += 1

        ratio = numeric_count / max(len(series), 1)
        if ratio >= 0.5:
            column_types[col] = "numeric"
        else:
            column_types[col] = "text"

    # Primary entity column: text column with highest uniqueness ratio
    best_entity_col = None
    best_uniqueness_ratio = -1.0

    for col in headers:
        if column_types.get(col) == "text":
            non_null_series = df[col].dropna()
            if len(non_null_series) > 0:
                uniqueness_ratio = non_null_series.nunique() / len(non_null_series)
                if uniqueness_ratio > best_uniqueness_ratio:
                    best_uniqueness_ratio = uniqueness_ratio
                    best_entity_col = col

    if not best_entity_col and headers:
        best_entity_col = headers[0]

    return {
        "sheet_name": sheet_name,
        "headers": headers,
        "column_types": column_types,
        "entity_column": best_entity_col
    }


def is_summary_sheet(sheet_name, df):
    """
    Structural summary sheet detection without hardcoded sheet names.
    Detects summary/legend/instruction sheets based on structural characteristics:
      - Low row count (< 2 rows)
      - Low data cell density (< 25% non-empty cells)
    """
    name_norm = normalize_string(sheet_name)
    words = set(re.findall(r"\b[a-z0-9]+\b", name_norm))

    if "summary" in words or "legend" in words or "instructions" in words:
        return True

    if len(df) < 2:
        return True

    total_cells = df.shape[0] * df.shape[1]
    if total_cells > 0:
        non_empty_cells = df.notna().sum().sum()
        density = non_empty_cells / total_cells
        if density < 0.25:
            return True

    return False


def load_spreadsheet_sheets(file_path):
    """
    Loads all sheets from an .xlsx, .xls, or .csv file into DataFrames and schemas.
    Uses openpyxl / pandas with generic header detection.
    """
    ext = os.path.splitext(file_path)[1].lower()
    sheets_dict = {}

    if ext in {".xlsx", ".xls"}:
        wb = load_workbook(file_path, data_only=True)
        for sheet in wb.worksheets:
            sheet_name = sheet.title.strip()

            values = list(sheet.iter_rows(values_only=True))
            if not values:
                continue

            header_idx = find_header_row_index(values)
            raw_headers = list(values[header_idx])

            # Deduplicate and clean headers
            headers = []
            seen = set()
            for idx, h in enumerate(raw_headers):
                h_str = str(h).strip() if h is not None and str(h).strip() else f"Col_{idx+1}"
                orig_h = h_str
                counter = 1
                while h_str.lower() in seen:
                    h_str = f"{orig_h}_{counter}"
                    counter += 1
                seen.add(h_str.lower())
                headers.append(h_str)

            rows = []
            for row_vals in values[header_idx + 1:]:
                row_dict = {}
                for idx, h_name in enumerate(headers):
                    val = row_vals[idx] if idx < len(row_vals) else None
                    row_dict[h_name] = val

                if is_valid_data_row(row_dict, headers):
                    rows.append(row_dict)

            if rows:
                df = pd.DataFrame(rows)
                if not is_summary_sheet(sheet_name, df):
                    schema = inspect_sheet_schema(df, sheet_name)
                    sheets_dict[sheet_name] = {
                        "df": df,
                        "schema": schema
                    }

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        sheet_name = os.path.basename(file_path)
        schema = inspect_sheet_schema(df, sheet_name)
        sheets_dict[sheet_name] = {
            "df": df,
            "schema": schema
        }

    return sheets_dict


def load_structured_spreadsheet_from_lancedb(file_path=None):
    """
    Loads structured spreadsheet rows from LanceDB and converts them
    into the same DataFrame-based representation used by the query engine.
    Falls back to direct file reading if LanceDB is empty or missing the file.
    """
    table = get_spreadsheet_table()

    if table is not None and table.count_rows() > 0:
        rows = table.search().limit(100000).to_list()

        if file_path:
            target = os.path.abspath(str(file_path)).lower()
            target_name = os.path.basename(target).lower()

            rows = [
                row for row in rows
                if (
                    os.path.abspath(str(row.get("source_path", ""))).lower() == target
                    or str(row.get("source_file", "")).lower() == target_name
                )
            ]

        if rows:
            grouped = {}

            for row in rows:
                try:
                    row_data = json.loads(row.get("row_data", "{}"))
                except Exception:
                    row_data = {}

                sheet_name = row.get("sheet_name", "Sheet1")

                if sheet_name not in grouped:
                    grouped[sheet_name] = []

                grouped[sheet_name].append({
                    **row_data,
                    "_sheet": sheet_name,
                    "_row_number": row.get("row_number"),
                    "_source_file": row.get("source_file"),
                    "_source_path": row.get("source_path"),
                })

            result = {}

            for sheet_name, records in grouped.items():
                if not records:
                    continue

                df = pd.DataFrame(records)

                # Remove internal metadata columns from actual spreadsheet schema
                data_headers = [c for c in df.columns if not str(c).startswith("_")]
                data_df = df[data_headers].copy()

                if not is_summary_sheet(sheet_name, data_df):
                    schema = inspect_sheet_schema(data_df, sheet_name)
                    result[sheet_name] = {
                        "df": data_df,
                        "schema": schema,
                    }

            if result:
                return result

    # Fallback: direct disk load if not present in LanceDB
    if file_path and os.path.exists(file_path):
        return load_spreadsheet_sheets(file_path)

    return {}


def find_applicable_spreadsheets(candidate_paths, question):
    """
    Inspects candidate spreadsheet files in watched_folder against user query tokens.
    Filters out generic media/table terms (e.g. 'list', 'show', 'files', 'data', 'records', 'items').
    Returns candidate paths sorted by match relevance, auto-resolving to a single workbook
    when it clearly outperforms other candidate workbooks.
    """
    q_norm = strip_file_mentions(normalize_string(question))
    query_tokens = [w for w in _words(q_norm) if w not in FILLER and len(w) > 2 and not re.fullmatch(r"\d+(\.\d+)?", w)]
    if not query_tokens:
        return candidate_paths

    scores = {}
    coverage = {}

    for path in candidate_paths:
        sheets = load_structured_spreadsheet_from_lancedb(path)
        if not sheets:
            continue

        score = 0
        covered = set()
        for sheet_name, data in sheets.items():
            s_norm = normalize_string(sheet_name)
            df = data["df"]
            schema = data["schema"]
            headers = schema["headers"]

            for token in query_tokens:
                if _canon(token) in {_canon(w) for w in _words(s_norm)}:
                    score += 10
                    covered.add(token)
                for col in headers:
                    if _canon(token) in {_canon(w) for w in _words(col)}:
                        score += 5
                        covered.add(token)
                    if schema["column_types"].get(col) == "text" and col in df.columns:
                        mask, _ = token_mask(df[col], token)
                        if mask.any():
                            score += 2
                            covered.add(token)

        scores[path] = score
        coverage[path] = len(covered)

    if not scores:
        return candidate_paths

    # A workbook that can ground MORE of the query words wins, even if another
    # workbook has a higher raw hit count (e.g. role words repeated on every row).
    best_cov = max(coverage.values())
    pool = [p for p in scores if coverage[p] == best_cov]
    pool.sort(key=lambda p: scores[p], reverse=True)
    top_score = scores[pool[0]]

    if best_cov > 0:
        # Same coverage => the question fits every workbook in the pool. Raw score depends on row counts,
        # so it must not silently eliminate a workbook. Caller answers each one (execute_across_spreadsheets).
        return pool
    return candidate_paths


def build_runtime_schema_json(sheets):
    """
    Constructs a JSON-encodable runtime schema representation of all data-bearing sheets,
    including column data types, row counts, and unique categorical values per text column.
    Used for prompting the LLM query planner and for deterministic grounding validation.
    """
    schema_summary = []
    for sheet_name, data in sheets.items():
        df = data["df"]
        schema = data["schema"]
        headers = schema["headers"]
        col_types = schema["column_types"]

        categorical_values = {}
        for col in headers:
            if col in df.columns:
                if col_types.get(col) == "text":
                    unique_vals = df[col].dropna().unique()[:50].tolist()
                    categorical_values[col] = [str(v) for v in unique_vals if str(v).strip()]
                else:
                    valid_nums = df[col].apply(parse_numeric_value).dropna()
                    if not valid_nums.empty:
                        categorical_values[col] = {
                            "min": float(valid_nums.min()),
                            "max": float(valid_nums.max()),
                            "sample": [float(v) for v in valid_nums.unique()[:5]]
                        }

        schema_summary.append({
            "sheet_name": sheet_name,
            "headers": headers,
            "column_types": col_types,
            "row_count": len(df),
            "categorical_values": categorical_values
        })

    return json.dumps(schema_summary, indent=2)


def generate_structured_query_plan(question, sheets, conversation_history="", structured_context=None):
    """
    Uses Ollama / LLM to convert natural human language and conversational history
    into a structured JSON Query Plan grounded strictly in the actual runtime schema.
    """
    schema_json = build_runtime_schema_json(sheets)
    context_str = json.dumps(structured_context or {}, indent=2)

    prompt = f"""You are a Natural Language to Structured Query Plan parser for spreadsheets.

USER QUESTION: "{question}"
CONVERSATIONAL HISTORY: "{conversation_history}"
STRUCTURED INTERACTION CONTEXT:
{context_str}

ACTUAL RUNTIME SPREADSHEET SCHEMA & DATA VALUES:
{schema_json}

Convert the user's question into a JSON object matching this EXACT schema:
{{
  "operation": "LIST" | "COUNT" | "GROUP_BY" | "RANKING" | "AGGREGATE" | "UNGROUNDED",
  "target_sheet": "<exact_sheet_name_from_schema_or_null>",
  "entity_column": "<exact_column_name_to_display_or_null>",
  "metric_column": "<exact_numeric_column_name_or_null>",
  "filters": [
    {{
      "column": "<exact_column_name>",
      "operator": "EQUALS" | "CONTAINS" | "GREATER_THAN" | "LESS_THAN" | "BETWEEN",
      "value": "<value_or_threshold>"
    }}
  ],
  "group_by_column": "<exact_column_name_or_sheet_name_or_null>",
  "aggregation": "COUNT" | "AVG" | "SUM" | "MIN" | "MAX" | "DISTINCT_COUNT" | "NONE",
  "sort_direction": "ASC" | "DESC" | "NONE",
  "limit": <number_or_null>,
  "ungrounded_reason": "<string_explaining_missing_column_or_value_or_null>"
}}

RULES:
1. Ground target_sheet, entity_column, metric_column, filters, and group_by_column strictly in the actual schema headers, sheet names, and categorical data values.
2. If the user query contains a filter term or concept that DOES NOT exist in any sheet name, column header, or categorical data value in the runtime schema, set operation="UNGROUNDED" and set ungrounded_reason to a description of the missing term.
3. If the user query is a follow-up or incomplete request, inherit target_sheet, entity_column, and filters from the STRUCTURED INTERACTION CONTEXT.
4. Return ONLY valid JSON.
"""

    sys_instruction = "You are a precise JSON query plan parser for spreadsheets. Output valid JSON only."

    try:
        raw_response = ask_spreadsheet_llm(prompt, system_instruction=sys_instruction)
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group(0))
            return plan
    except Exception:
        pass

    return None


def generate_spreadsheet_summary(file_path, sheets, question=""):
    """
    Generates a comprehensive statistical and natural-language summary overview of a spreadsheet dataset.
    Profiles sheet names, row counts, column types, metric ranges, top category distributions,
    and uses the dedicated SPREADSHEET_MODEL_NAME to synthesize a structured markdown summary.
    """
    if not sheets:
        return f"No readable data found in '{os.path.basename(file_path)}'.", 0

    filename = os.path.basename(file_path)
    total_rows_across_sheets = 0
    all_sheet_reports = []

    for sheet_name, sheet_data in sheets.items():
        df = sheet_data["df"]
        schema = sheet_data["schema"]
        headers = schema["headers"]
        col_types = schema["column_types"]
        row_count = len(df)
        total_rows_across_sheets += row_count

        sheet_report = f"### Sheet: '{sheet_name}' ({row_count} total records)\n"
        sheet_report += f"- **Columns ({len(headers)}):** {', '.join(headers)}\n"

        # Categorical column profiling (top 3 values with counts)
        text_cols = [c for c in headers if col_types.get(c) == "text"]
        cat_insights = []
        for c in text_cols[:4]:
            val_counts = df[c].dropna().value_counts()
            if not val_counts.empty:
                top_items = [f"{val} ({cnt})" for val, cnt in val_counts.head(3).items()]
                cat_insights.append(f"  - **{c}**: Top values: {', '.join(top_items)}")
        if cat_insights:
            sheet_report += "- **Key Category Breakdown:**\n" + "\n".join(cat_insights) + "\n"

        # Numeric metric profiling (min, max, avg)
        num_cols = [c for c in headers if col_types.get(c) == "numeric"]
        num_insights = []
        for c in num_cols[:3]:
            parsed_vals = df[c].apply(parse_numeric_value).dropna()
            if not parsed_vals.empty:
                min_v = parsed_vals.min()
                max_v = parsed_vals.max()
                avg_v = parsed_vals.mean()
                num_insights.append(f"  - **{c}**: Range = {min_v:g} to {max_v:g} (Average: {avg_v:.2f})")
        if num_insights:
            sheet_report += "- **Metric Ranges:**\n" + "\n".join(num_insights) + "\n"

        all_sheet_reports.append(sheet_report)

    combined_profile = "\n".join(all_sheet_reports)

    # Use ask_spreadsheet_llm to synthesize a clean executive summary
    prompt = f"""
You are an expert Data Engineering and Analytics assistant.
Provide a clear, structured summary overview of the following spreadsheet file: '{filename}'.

DATASET STATISTICAL PROFILE:
{combined_profile}

Write a concise summary report highlighting:
1. What dataset this spreadsheet contains (purpose/domain).
2. Key structure (sheets, row count, main columns).
3. Significant insights (top categories, metric ranges).
Keep the summary factual, professional, and formatted in clean GitHub markdown.
"""
    sys_instruction = "You are a helpful spreadsheet analytics assistant. Synthesize grounded dataset summaries."

    try:
        llm_summary = ask_spreadsheet_llm(prompt, system_instruction=sys_instruction)
        final_answer = f"## Spreadsheet Overview: `{filename}`\n\n{combined_profile}\n### Executive Summary\n{llm_summary}"
    except Exception:
        final_answer = f"## Spreadsheet Overview: `{filename}`\n\n{combined_profile}"

    return final_answer, total_rows_across_sheets


def execute_spreadsheet_query(file_path, question):
    """
    Main domain-agnostic orchestrator for structured spreadsheet queries.
    Provides runtime schema to LLM to generate structured query plan,
    validates the plan deterministically, executes pandas operations,
    and returns complete grounded results.
    """
    sheets = load_structured_spreadsheet_from_lancedb(file_path)
    if not sheets:
        return f"Could not extract data from '{os.path.basename(file_path)}'.", 0

    q_norm = normalize_string(question)
    last_interaction = get_last_interaction()
    conv_history = last_interaction.get("question", "") if last_interaction else ""
    struct_ctx = get_spreadsheet_context()

    # Check for Summary / Overview intent
    is_summary_op = bool(re.search(r"\b(?:summarize|summarise|summary|overview|describe|statistics|structure)\b", q_norm))
    if is_summary_op:
        return generate_spreadsheet_summary(file_path, sheets, question)

    # Generate LLM Query Plan
    llm_plan = generate_structured_query_plan(
        question,
        sheets,
        conversation_history=conv_history,
        structured_context=struct_ctx
    )

    if llm_plan and llm_plan.get("operation") in ["SUMMARY", "OVERVIEW"]:
        return generate_spreadsheet_summary(file_path, sheets, question)

    # Fail closed if LLM query plan determined ungrounded concept
    if llm_plan and llm_plan.get("operation") == "UNGROUNDED":
        reason = llm_plan.get("ungrounded_reason") or f"unsupported filter in '{question}'"
        return f"I can't determine that from this spreadsheet because {reason}.", 0


    # Sheet scope: restrict ONLY to sheets the question names; otherwise search ALL sheets.
    named_sheets = []
    for sheet_name in sheets.keys():
        toks = [w for w in normalize_string(sheet_name).split() if len(w) > 1]
        if any(re.search(r"\b" + re.escape(t) + r"\b", q_norm) for t in toks):
            named_sheets.append(sheet_name)

    scope_sheets = named_sheets or list(sheets.keys())
    sheet_col = "Sheet" if "Sheet" not in sheets[scope_sheets[0]]["df"].columns else "_Sheet"

    if len(scope_sheets) == 1:
        target_sheet_name = scope_sheets[0]
        df = sheets[target_sheet_name]["df"].copy()
        multi_scope = False
    else:
        frames = []
        for sn in scope_sheets:
            f = sheets[sn]["df"].copy()
            f[sheet_col] = sn
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        target_sheet_name = "all sheets" if not named_sheets else ", ".join(scope_sheets)
        multi_scope = True

    schema = inspect_sheet_schema(df, target_sheet_name)
    from collections import Counter
    _ents = Counter(sheets[sn]["schema"]["entity_column"] for sn in scope_sheets)
    preferred_entity = _ents.most_common(1)[0][0]
    headers = schema["headers"]
    col_types = schema["column_types"]
    if multi_scope:
        col_types[sheet_col] = "text"

    # Dynamic Column Resolution
    entity_col = None
    if llm_plan and llm_plan.get("entity_column") in headers:
        entity_col = llm_plan.get("entity_column")
    if not entity_col and preferred_entity in headers:
        entity_col = preferred_entity
    if not entity_col:
        best_ratio = -1.0
        for col in headers:
            if col_types.get(col) == "text":
                non_nulls = df[col].dropna()
                if len(non_nulls) > 0:
                    ratio = non_nulls.nunique() / len(non_nulls)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        entity_col = col
    if not entity_col:
        entity_col = headers[0]

    metric_col = None
    if llm_plan and llm_plan.get("metric_column") in headers:
        metric_col = llm_plan.get("metric_column")
    if not metric_col:
        numeric_cols = [c for c in headers if col_types.get(c) == "numeric"]
        if numeric_cols:
            for c in numeric_cols:
                c_norm = normalize_string(c)
                if any(w in q_norm for w in c_norm.split() if len(w) > 2):
                    metric_col = c
                    break
            if not metric_col:
                metric_col = numeric_cols[0]

    if entity_col and entity_col in df.columns:
        df = df[df[entity_col].notna() & (df[entity_col].astype(str).str.strip() != "")].reset_index(drop=True)

    # ------------------------------------------------------------
    # GROUP BY / AGGREGATION ENGINE
    # ------------------------------------------------------------
    group_phrase_match = re.search(
        r"\b(?:in\s+each|per|by|for\s+each|grouped?\s+by)\s+([a-z0-9\s/_\-]+)",
        q_norm
    )
    is_group_op = (llm_plan and llm_plan.get("operation") == "GROUP_BY") or bool(group_phrase_match)

    if is_group_op:
        combined_list = []
        for s_name, d in sheets.items():
            s_df = d["df"].copy()
            s_df["_sheet_name"] = s_name
            combined_list.append(s_df)

        combined_df = pd.concat(combined_list, ignore_index=True).reset_index(drop=True)
        u_headers = [c for c in combined_df.columns if not str(c).startswith("_")]

        group_col = None
        if llm_plan and llm_plan.get("group_by_column"):
            grp = llm_plan.get("group_by_column")
            if grp in u_headers or grp == "_sheet_name":
                group_col = grp

        if not group_col and group_phrase_match:
            raw_phrase = group_phrase_match.group(1).strip()
            g_tokens = [w for w in raw_phrase.split() if len(w) > 2]
            for col in u_headers:
                c_norm = normalize_string(col)
                if any(re.search(r"\b" + re.escape(t) + r"\b", c_norm) for t in g_tokens):
                    group_col = col
                    break
            if not group_col and len(sheets) > 1:
                group_col = "_sheet_name"

        if not group_col:
            group_col = "_sheet_name" if len(sheets) > 1 else u_headers[0]

        eval_df = combined_df[combined_df[group_col].notna() & (combined_df[group_col].astype(str).str.strip() != "")].reset_index(drop=True)

        is_avg = bool(re.search(r"\b(?:average|avg|mean)\b", q_norm))
        is_sum = bool(re.search(r"\b(?:total|sum)\b", q_norm))
        is_ranking = bool(re.search(r"\b(?:highest|lowest|top|bottom|max|min|best|worst|largest|smallest)\b", q_norm))

        match_top = re.search(r"\btop\s+(\d+)\b", q_norm)
        limit = int(match_top.group(1)) if match_top else None

        descending = not bool(re.search(r"\b(?:lowest|smallest|minimum|min|bottom|least|cheapest)\b", q_norm))

        if is_avg or is_sum or (is_ranking and metric_col):
            eval_df["_numeric_val"] = eval_df[metric_col].apply(parse_numeric_value)
            eval_df = eval_df.dropna(subset=["_numeric_val"]).reset_index(drop=True)

            if is_avg:
                grouped = eval_df.groupby(group_col)["_numeric_val"].mean()
            elif is_sum:
                grouped = eval_df.groupby(group_col)["_numeric_val"].sum()
            else:
                grouped = eval_df.groupby(group_col)["_numeric_val"].mean()

            if is_ranking:
                grouped = grouped.sort_values(ascending=not descending)
            if limit:
                grouped = grouped.head(limit)

            lines = []
            for idx, (grp_name, val) in enumerate(grouped.items(), 1):
                lines.append(f"{idx}. {grp_name}: {val:.2f}" if is_avg else f"{idx}. {grp_name}: {val:g}")
            return "\n".join(lines), len(lines)

        else:
            if entity_col and entity_col in eval_df.columns:
                grouped = eval_df.groupby(group_col)[entity_col].nunique()
            else:
                grouped = eval_df.groupby(group_col).size()

            if is_ranking:
                grouped = grouped.sort_values(ascending=not descending)
            if limit:
                grouped = grouped.head(limit)

            lines = []
            for idx, (grp_name, cnt) in enumerate(grouped.items(), 1):
                lines.append(f"{idx}. {grp_name}: {cnt}")
            return "\n".join(lines), len(lines)

    # ------------------------------------------------------------
    # FILTER CONDITIONS & GROUNDING (FAIL CLOSED)
    # ------------------------------------------------------------
    filtered_df = df.copy()
    ungrounded_phrases = []

    q_filter_text = q_norm
    if file_path:
        fname = os.path.basename(file_path).lower()
        fstem = os.path.splitext(fname)[0].lower()
        q_filter_text = q_filter_text.replace(fname, "").replace(fstem, "")
        for fn_word in re.findall(r"\b[a-z0-9]+\b", fstem):
            if len(fn_word) > 2:
                q_filter_text = re.sub(r"\b" + re.escape(fn_word) + r"\b", "", q_filter_text)

    q_filter_text = strip_file_mentions(q_norm, file_path)
    classified = classify_words(q_filter_text, headers, list(sheets.keys()))

    matched_tokens = {}      # unit (word or phrase) -> {column: (mask, mode)}
    soft_in_q = {w for w, k in classified if k == "soft"}

    def column_hits(words, phrase):
        colmap = {}
        for col in headers:
            if col_types.get(col) == "text" and col in df.columns:
                if phrase:
                    m, mode = phrase_mask(df[col], words), "phrase"
                else:
                    m, mode = token_mask(df[col], words[0])
                if m.any():
                    colmap[col] = (m, mode)
        return colmap

    def phrase_exists_anywhere(words):
        for d in sheets.values():
            sdf, sch = d["df"], d["schema"]
            for col in sch["headers"]:
                if sch["column_types"].get(col) == "text" and col in sdf.columns and phrase_mask(sdf[col], words).any():
                    return True
        return False

    for run in content_runs(classified):
        candidates = [run]
        hard_only = [w for w in run if w not in soft_in_q]
        if hard_only != run and hard_only:
            candidates.append(hard_only)                      # retry without soft words ('show data' filler)
        done = False
        for words in candidates:
            if len(words) >= 2:
                cm = column_hits(words, phrase=True)
                if cm:
                    matched_tokens[" ".join(words)] = cm
                    done = True
                    break
        if done:
            continue
        # The phrase exists in this workbook but not inside the requested scope (e.g. 'data engineer' in Indore):
        # the honest answer is "none". Do NOT loosen it into 'data' in one column AND 'engineer' in another.
        if any(len(w) >= 2 and phrase_exists_anywhere(w) for w in candidates):
            return f"No matching spreadsheet records found for '{' '.join(candidates[0])}' in {target_sheet_name}.", 0
        for token in candidates[-1]:                          # no contiguous phrase: each word must ground alone
            cm = column_hits([token], phrase=False)
            if cm:
                matched_tokens[token] = cm
            elif token not in soft_in_q:
                return (f"I can't determine that from this spreadsheet because I couldn't find a column or "
                        f"categorical value that represents '{token}' in the workbook."), 0

    matched_columns_and_vals = [(c, t) for t, cm in matched_tokens.items() for c in cm]
    loose_notes = []

    # every unit must match (AND); a unit may match in any column (OR)
    for key, colmap in matched_tokens.items():
        combined = None
        for col, (mask, mode) in colmap.items():
            combined = mask if combined is None else (combined | mask)
        filtered_df = filtered_df[combined.loc[filtered_df.index]]
    filtered_df = filtered_df.reset_index(drop=True)

    # Apply LLM plan filters if available
    if llm_plan and llm_plan.get("filters") and not matched_tokens:
        for f_item in llm_plan.get("filters"):
            f_col = f_item.get("column")
            f_op = f_item.get("operator")
            f_val = f_item.get("value")
            if f_col in filtered_df.columns and f_val is not None:
                val_clean = str(f_val).lower().strip()
                if target_sheet_name and val_clean == target_sheet_name.lower():
                    continue
                if f_op in ("EQUALS", "CONTAINS"):
                    mask = filtered_df[f_col].astype(str).str.lower().str.contains(re.escape(val_clean), regex=True, na=False)
                    mask = mask.reindex(filtered_df.index, fill_value=False)
                    filtered_df = filtered_df[mask].reset_index(drop=True)

    # Numeric threshold comparison filter
    if metric_col and metric_col in filtered_df.columns:
        filtered_df["_numeric_val"] = filtered_df[metric_col].apply(parse_numeric_value)
        num_match = re.search(r"\b(?:above|over|greater\s+than|more\s+than|>)\s+(\d+(?:\.\d+)?)\b", q_norm)
        if num_match:
            threshold = float(num_match.group(1))
            filtered_df = filtered_df[filtered_df["_numeric_val"] > threshold].reset_index(drop=True)

    # ------------------------------------------------------------
    # 6. Structured Operations Execution (COUNT, AVG, SUM, SORT, LIST)
    # ------------------------------------------------------------
    is_count = (llm_plan and llm_plan.get("operation") == "COUNT") or bool(re.search(r"\b(?:how\s+many|number\s+of|count|is\s+there\s+any)\b", q_norm))
    is_avg = bool(re.search(r"\b(?:average|avg|mean)\b", q_norm))
    is_sum = bool(re.search(r"\b(?:total|sum)\b", q_norm)) and not is_count
    is_ranking = (llm_plan and llm_plan.get("operation") == "RANKING") or bool(re.search(r"\b(?:highest|lowest|top|bottom|max|min|best|worst|largest|smallest)\b", q_norm))

    if is_count and not is_ranking and not is_avg and not is_sum:
        if entity_col and entity_col in filtered_df.columns:
            cnt = filtered_df[entity_col].nunique()
        else:
            cnt = len(filtered_df)

        rows_n = len(filtered_df)
        dup_note = f" ({rows_n} rows; some {entity_col} values repeat across sheets)" if rows_n != cnt else ""
        if matched_columns_and_vals:
            filter_desc = ", ".join([f"{c}='{v}'" for c, v in matched_columns_and_vals])
            return f"There are {cnt} items matching {filter_desc} in '{target_sheet_name}'.{dup_note}", cnt
        return f"There are {cnt} items listed in '{target_sheet_name}'.{dup_note}", cnt

    if is_avg and metric_col and "_numeric_val" in filtered_df.columns:
        valid_nums = filtered_df["_numeric_val"].dropna()
        if not valid_nums.empty:
            avg_val = valid_nums.mean()
            return f"The average {metric_col} is {avg_val:.2f}.", len(valid_nums)

    if is_sum and metric_col and "_numeric_val" in filtered_df.columns:
        valid_nums = filtered_df["_numeric_val"].dropna()
        if not valid_nums.empty:
            sum_val = valid_nums.sum()
            return f"The total {metric_col} is {sum_val:g}.", len(valid_nums)

    if metric_col and is_ranking:
        descending = True
        if re.search(r"\b(?:lowest|smallest|minimum|min|bottom|least|cheapest)\b", q_norm):
            descending = False

        limit = None
        match_top = re.search(r"\btop\s+(\d+)\b", q_norm)
        if match_top:
            limit = int(match_top.group(1))

        if not limit and llm_plan and llm_plan.get("limit"):
            limit = llm_plan.get("limit")

        if "_numeric_val" not in filtered_df.columns:
            filtered_df["_numeric_val"] = filtered_df[metric_col].apply(parse_numeric_value)

        df_sorted = filtered_df.dropna(subset=["_numeric_val"]).sort_values(by="_numeric_val", ascending=not descending)
        if limit:
            df_res = df_sorted.head(limit)
        else:
            df_res = df_sorted

        if not df_res.empty:
            lines = []
            for idx, (_, row) in enumerate(df_res.iterrows(), 1):
                ent_val = str(row.get(entity_col, "")).strip() if entity_col else f"Item {idx}"
                met_val = str(row.get(metric_col, "")).strip()
                lines.append(f"{idx}. {entity_col}: {ent_val} | {metric_col}: {met_val}")
            return "\n".join(lines), len(df_res)

    if not filtered_df.empty:
        lines = []
        display_cols = [entity_col] if entity_col in filtered_df.columns else []
        if multi_scope and sheet_col in filtered_df.columns:
            display_cols.append(sheet_col)
        for c, _ in matched_columns_and_vals:            # show the columns the answer was filtered on
            if c in filtered_df.columns and c not in display_cols:
                display_cols.append(c)
        metric_asked = metric_col and any(w in q_norm for w in normalize_string(metric_col).split() if len(w) > 2)
        if metric_asked and metric_col in filtered_df.columns and metric_col not in display_cols:
            display_cols.append(metric_col)
        if not display_cols:
            display_cols = headers[:3]

        # NO TRUNCATION: Format ALL matching rows unless limit was explicitly requested
        match_limit = re.search(r"\b(?:top|first|head|limit)\s+(\d+)\b", q_norm)
        if match_limit:
            user_limit = int(match_limit.group(1))
            eval_rows = filtered_df.head(user_limit)
        elif llm_plan and llm_plan.get("limit"):
            eval_rows = filtered_df.head(llm_plan.get("limit"))
        else:
            eval_rows = filtered_df

        for idx, (_, row) in enumerate(eval_rows.iterrows(), 1):
            parts = [f"{col}: {row.get(col, '')}" for col in display_cols if pd.notna(row.get(col))]
            lines.append(f"{idx}. " + " | ".join(parts))

        if loose_notes:
            lines.append(f"(Note: matched {', '.join(repr(t) for t in loose_notes)} loosely against abbreviations such as 'Dev'.)")
        return "\n".join(lines), len(filtered_df)

    return "No matching spreadsheet records found.", 0


# Backward Compatibility Bridges for Legacy Callers
def load_spreadsheet_rows(path):
    sheets = load_spreadsheet_sheets(path)
    rows = []
    for sheet_name, data in sheets.items():
        df = data["df"]
        for _, row in df.iterrows():
            rec = dict(row)
            rec["_sheet"] = sheet_name
            rows.append(rec)
    return rows


def filter_rows(rows, question):
    return rows


def sort_rows(rows, question):
    return rows


def group_by_city(rows):
    return {}


def format_rows(rows):
    if not rows:
        return "No matching rows found."
    headers = list(rows[0].keys())
    lines = []
    for idx, row in enumerate(rows, 1):
        parts = [f"{col}: {row[col]}" for col in headers[:3] if col != "_sheet"]
        lines.append(f"{idx}. " + " | ".join(parts))
    return "\n".join(lines)


def requested_result_limit(question):
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", question.lower())
    if match:
        return int(match.group(1))
    if re.search(r"\b(?:highest|lowest|largest|smallest|max|min)\b", question.lower()):
        return 1
    return None


def format_ranking_result(rows, question):
    return format_rows(rows)


def execute_across_spreadsheets(paths, question):
    """
    Use when several workbooks tie for the same question and the user did not name one.
    Answers each workbook that can ground the question and labels the answers by file.
    """
    parts, total, first_failure = [], 0, None
    for p in paths:
        answer, n = execute_spreadsheet_query(p, question)
        if answer.startswith("I can't determine") or answer.startswith("Could not extract"):
            first_failure = first_failure or answer
            continue
        parts.append(f"### {os.path.basename(p)}\n{answer}")
        total += n
    if not parts:
        return first_failure or "No matching spreadsheet records found.", 0
    return "\n\n".join(parts), total
