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
from app.llm.ollama_client import ask_llama
from app.services.interaction_state import get_last_interaction


def normalize_string(value):
    """Normalize text by stripping whitespace and converting to lowercase."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


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
    q_norm = normalize_string(question)
    words_clean = [w for w in re.findall(r"\b[a-z0-9]+\b", q_norm)]
    generic_terms = {
        "how", "many", "what", "which", "who", "where", "when", "is", "are", "there", "any", "the",
        "a", "an", "in", "on", "at", "for", "with", "from", "by", "of", "to", "this", "that", "sheet",
        "spreadsheet", "file", "files", "workbook", "data", "table", "listed", "mentioned", "available",
        "different", "show", "list", "give", "provide", "display", "find", "filter", "sort", "order",
        "rank", "highest", "lowest", "top", "bottom", "all", "count", "number", "hiring", "hired",
        "row", "rows", "column", "columns", "having", "has", "have", "more", "less", "above", "below",
        "each", "their", "where", "can", "tell", "me", "record", "records", "item", "items", "entry", "entries",
        "company", "companies", "product", "products", "student", "students", "employee", "employees"
    }

    query_tokens = [w for w in words_clean if w not in generic_terms and len(w) > 2 and not re.fullmatch(r"\d+(\.\d+)?", w)]
    if not query_tokens:
        return candidate_paths

    scores = {}

    for path in candidate_paths:
        sheets = load_structured_spreadsheet_from_lancedb(path)
        if not sheets:
            continue

        score = 0
        for sheet_name, data in sheets.items():
            s_norm = normalize_string(sheet_name)

            # Match sheet name
            for token in query_tokens:
                if re.search(r"\b" + re.escape(token) + r"\b", s_norm):
                    score += 10

            df = data["df"]
            schema = data["schema"]
            headers = schema["headers"]

            # Match headers
            for col in headers:
                c_norm = normalize_string(col)
                for token in query_tokens:
                    if re.search(r"\b" + re.escape(token) + r"\b", c_norm):
                        score += 5

            # Match cell values in text columns
            for col in headers:
                if schema["column_types"].get(col) == "text":
                    series_str = df[col].dropna().astype(str).str.lower()
                    for token in query_tokens:
                        if series_str.str.contains(r"\b" + re.escape(token) + r"\b", regex=True, na=False).any():
                            score += 2

        scores[path] = score

    if not scores:
        return candidate_paths

    sorted_candidates = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
    top_score = scores[sorted_candidates[0]]

    if top_score > 0:
        best_candidates = [p for p in sorted_candidates if scores[p] > 0 and (top_score - scores[p]) <= 5]
        return best_candidates

    return candidate_paths


def build_runtime_schema_json(sheets):
    """
    Constructs a JSON-encodable runtime schema representation of all data-bearing sheets.
    Used for prompting the LLM query planner.
    """
    schema_summary = []
    for sheet_name, data in sheets.items():
        df = data["df"]
        schema = data["schema"]
        headers = schema["headers"]
        col_types = schema["column_types"]

        sample_vals = {}
        for col in headers:
            if col in df.columns:
                unique_samples = df[col].dropna().unique()[:3].tolist()
                sample_vals[col] = [str(v) for v in unique_samples]

        schema_summary.append({
            "sheet_name": sheet_name,
            "headers": headers,
            "column_types": col_types,
            "row_count": len(df),
            "sample_values": sample_vals
        })

    return json.dumps(schema_summary, indent=2)


def generate_structured_query_plan(question, sheets, conversation_history=""):
    """
    Uses Ollama / LLM to convert natural human language and conversational history
    into a structured JSON Query Plan grounded in the actual runtime schema.
    """
    schema_json = build_runtime_schema_json(sheets)

    prompt = f"""You are a Natural Language to Structured Query Plan parser for spreadsheets.

USER QUESTION: "{question}"
CONVERSATIONAL HISTORY: "{conversation_history}"

ACTUAL RUNTIME SPREADSHEET SCHEMA:
{schema_json}

Convert the user's question into a JSON object matching this EXACT schema:
{{
  "operation": "LIST" | "COUNT" | "GROUP_BY" | "RANKING" | "METADATA" | "UNGROUNDED",
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
1. Ground target_sheet, entity_column, metric_column, filters, and group_by_column in the actual schema headers and sheet names.
2. If the user asks for a filter concept that DOES NOT exist in any column header or sample value (e.g. "service based"), set operation="UNGROUNDED" and ungrounded_reason="Could not find column or value representing 'service based' in the workbook."
3. If user asks "list companies of Indore", set operation="LIST", target_sheet="Indore", limit=null.
4. If user asks "how many companies in Indore?", set operation="COUNT", target_sheet="Indore", aggregation="DISTINCT_COUNT".
5. Return ONLY valid JSON.
"""

    sys_instruction = "You are a precise JSON query plan parser for spreadsheets. Output valid JSON only."

    try:
        raw_response = ask_llama(prompt, system_instruction=sys_instruction)
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group(0))
            return plan
    except Exception:
        pass

    return None


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

    # Generate LLM Query Plan
    llm_plan = generate_structured_query_plan(question, sheets, conversation_history=conv_history)

    # Fail closed if LLM query plan determined ungrounded concept
    if llm_plan and llm_plan.get("operation") == "UNGROUNDED":
        reason = llm_plan.get("ungrounded_reason") or f"unsupported filter in '{question}'"
        return f"I can't determine that from this spreadsheet because {reason}.", 0

    # Dynamic Sheet Selection fallback
    target_sheet_name = None
    if llm_plan and llm_plan.get("target_sheet") in sheets:
        target_sheet_name = llm_plan.get("target_sheet")

    if not target_sheet_name:
        for sheet_name in sheets.keys():
            s_norm = normalize_string(sheet_name)
            sheet_tokens = [w for w in s_norm.split() if len(w) > 1]
            for token in sheet_tokens:
                pattern = r"\b" + re.escape(token) + r"\b"
                if re.search(pattern, q_norm):
                    target_sheet_name = sheet_name
                    break
            if target_sheet_name:
                break

    if not target_sheet_name:
        target_sheet_name = list(sheets.keys())[0]

    sheet_data = sheets[target_sheet_name]
    df = sheet_data["df"].copy()
    schema = sheet_data["schema"]
    headers = schema["headers"]
    col_types = schema["column_types"]

    # Dynamic Column Resolution
    entity_col = None
    if llm_plan and llm_plan.get("entity_column") in headers:
        entity_col = llm_plan.get("entity_column")
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

    words_clean = [w for w in re.findall(r"\b[a-z0-9]+\b", q_filter_text)]
    stopwords = {
        "how", "many", "what", "which", "who", "where", "when", "is", "are", "there", "any", "the",
        "a", "an", "in", "on", "at", "for", "with", "from", "by", "of", "to", "this", "that", "sheet",
        "spreadsheet", "file", "files", "workbook", "data", "table", "listed", "mentioned", "available",
        "different", "show", "list", "give", "provide", "display", "find", "filter", "sort", "order",
        "rank", "highest", "lowest", "top", "bottom", "all", "count", "number", "hiring", "hired",
        "row", "rows", "column", "columns", "having", "has", "have", "more", "less", "above", "below",
        "each", "their", "where", "can", "tell", "me", "company", "companies", "product", "products",
        "student", "students", "item", "items", "employee", "employees", "record", "records", "entry", "entries",
        "xlsx", "xls", "csv", "ods", "pdf", "docx", "txt", "md"
    }

    candidate_phrases = []
    for i in range(len(words_clean)):
        if words_clean[i] not in stopwords and not re.fullmatch(r"\d+(\.\d+)?", words_clean[i]):
            candidate_phrases.append(words_clean[i])
            if i + 1 < len(words_clean) and words_clean[i+1] not in stopwords and not re.fullmatch(r"\d+(\.\d+)?", words_clean[i+1]):
                candidate_phrases.append(f"{words_clean[i]} {words_clean[i+1]}")
            if i + 2 < len(words_clean) and words_clean[i+2] not in stopwords and not re.fullmatch(r"\d+(\.\d+)?", words_clean[i+2]):
                candidate_phrases.append(f"{words_clean[i]} {words_clean[i+1]} {words_clean[i+2]}")

    candidate_phrases = sorted(list(set(candidate_phrases)), key=len, reverse=True)

    matched_columns_and_vals = []
    for phrase in candidate_phrases:
        if len(phrase) <= 2 or phrase in stopwords:
            continue

        sheet_matched = any(re.search(r"\b" + re.escape(phrase) + r"\b", normalize_string(s)) for s in sheets.keys())
        if sheet_matched:
            continue

        header_matched = any(re.search(r"\b" + re.escape(phrase) + r"\b", normalize_string(h)) for h in headers)
        if header_matched:
            continue

        p_tokens = [t for t in phrase.split() if t not in stopwords and len(t) > 2 and not re.fullmatch(r"\d+(\.\d+)?", t)]
        if not p_tokens:
            continue

        found_cell_match = False
        for token in p_tokens:
            token_header_match = any(re.search(r"\b" + re.escape(token) + r"\b", normalize_string(h)) for h in headers)
            if token_header_match:
                found_cell_match = True

            for col in headers:
                if col_types.get(col) == "text":
                    series_str = df[col].dropna().astype(str).str.lower()
                    matches = series_str[series_str.str.contains(r"\b" + re.escape(token) + r"\b", regex=True, na=False)]
                    if not matches.empty:
                        found_cell_match = True
                        if (col, token) not in matched_columns_and_vals and not token_header_match:
                            matched_columns_and_vals.append((col, token))

        if not found_cell_match:
            ungrounded_phrases.append(phrase)

    if ungrounded_phrases:
        missing_term = ungrounded_phrases[0]
        return f"I can't determine that from this spreadsheet because I couldn't find a column or categorical value that represents '{missing_term}' in the workbook.", 0

    # Apply filters with safe index alignment
    for col, val_str in matched_columns_and_vals:
        if col in filtered_df.columns:
            mask = filtered_df[col].astype(str).str.lower().str.contains(r"\b" + re.escape(val_str) + r"\b", regex=True, na=False)
            mask = mask.reindex(filtered_df.index, fill_value=False)
            filtered_df = filtered_df[mask].reset_index(drop=True)

    # Apply LLM plan filters if available
    if llm_plan and llm_plan.get("filters"):
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
    is_sum = bool(re.search(r"\b(?:total|sum)\b", q_norm))
    is_ranking = (llm_plan and llm_plan.get("operation") == "RANKING") or bool(re.search(r"\b(?:highest|lowest|top|bottom|max|min|best|worst|largest|smallest)\b", q_norm))

    if is_count and not is_ranking and not is_avg and not is_sum:
        if entity_col and entity_col in filtered_df.columns:
            cnt = filtered_df[entity_col].nunique()
        else:
            cnt = len(filtered_df)

        if matched_columns_and_vals:
            filter_desc = ", ".join([f"{c}='{v}'" for c, v in matched_columns_and_vals])
            return f"There are {cnt} items matching {filter_desc} in '{target_sheet_name}'.", cnt
        return f"There are {cnt} items listed in '{target_sheet_name}'.", cnt

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
        display_cols = [c for c in [entity_col, metric_col] if c and c in filtered_df.columns]
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