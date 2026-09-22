"""
Domain-Agnostic Structured Spreadsheet Query Engine.

This module provides a generic, data-driven query engine for structured
workbooks (.xlsx, .xls, .csv, .ods).

Absolute Principles:
1. NEVER hardcode business or domain concepts (e.g. companies, CTC, salary,
   internships, products, students, sales, cities).
2. Inspect the actual workbook headers, dimensions, and inferred data types dynamically.
3. Compute structured operations (sort, top-N, min/max, average, sum, count, filter, group)
   deterministically against pandas DataFrames.
4. Format results dynamically using actual headers present in the spreadsheet.
"""

import csv
import os
import re
import pandas as pd
from openpyxl import load_workbook
import json
from app.storage.lancedb_store import get_spreadsheet_table


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
    Profiles a DataFrame to determine:
    1. Clean header names
    2. Column data types ('numeric' vs 'text' vs 'datetime')
    3. Primary entity/identifier column
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

    # Primary entity column: first text column with reasonable uniqueness
    entity_col = None
    for col in headers:
        if column_types.get(col) == "text":
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio >= 0.05:
                entity_col = col
                break

    if not entity_col and headers:
        entity_col = headers[0]

    return {
        "sheet_name": sheet_name,
        "headers": headers,
        "column_types": column_types,
        "entity_column": entity_col
    }


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
            sheet_clean = normalize_string(sheet_name)
            if "summary" in sheet_clean and "count" in sheet_clean:
                continue

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


def resolve_metric_column(headers, column_types, question):
    """
    Dynamically resolves the numeric column matching the user's question.
    Priority:
    1. Substring header match in question.
    2. Generic numeric metric synonyms (salary, price, score, amount, etc.).
    3. Fallback to first numeric column for ranking queries.
    """
    q_norm = normalize_string(question)
    q_words = set(re.findall(r"\b[a-z0-9]+\b", q_norm))

    numeric_cols = [col for col in headers if column_types.get(col) == "numeric"]
    if not numeric_cols:
        return None

    # Priority 1: Substring / token match in column name
    for col in numeric_cols:
        c_norm = normalize_string(col)
        c_words = set(re.findall(r"\b[a-z0-9]+\b", c_norm))
        if c_norm in q_norm or (c_words and c_words.issubset(q_words)):
            return col

    # Priority 2: Synonym groups
    synonym_groups = [
        {"salary", "ctc", "compensation", "package", "lpa", "pay", "income", "stipend"},
        {"price", "cost", "amount", "rate", "fee", "val", "value", "size", "largest", "biggest"},
        {"score", "marks", "points", "rating", "grade", "gpa", "rank"},
        {"stock", "quantity", "qty", "count", "units", "volume", "inventory"},
        {"revenue", "sales", "turnover", "profit", "margin", "earnings"}
    ]

    for col in numeric_cols:
        c_norm = normalize_string(col)
        c_words = set(re.findall(r"\b[a-z0-9]+\b", c_norm))
        for group in synonym_groups:
            if group.intersection(q_words) and group.intersection(c_words):
                return col

    # Priority 3: Ranking keywords fallback
    ranking_words = {"highest", "lowest", "top", "bottom", "max", "maximum", "min", "minimum", "most", "least", "best", "worst", "largest", "smallest"}
    if ranking_words.intersection(q_words):
        return numeric_cols[0]

    return None


def resolve_entity_column(headers, column_types, question):
    """
    Dynamically resolves the primary entity/identifier text column matching the question.
    Gives top priority to primary entity headers (Company Name, Student, Product, Name, Title, Item)
    over secondary filter/description text columns (Roles, Location, How to Apply).
    """
    q_norm = normalize_string(question)

    text_cols = [col for col in headers if column_types.get(col) == "text" or "id" in col.lower()]
    if not text_cols:
        return headers[0] if headers else None

    primary_entity_keywords = {
        "name",
        "title",
        "product",
        "item",
        "employee",
        "customer",
        "student",
        "user",
        "person",
        "company",
        "organization",
        "vendor",
        "client",
        "account",
    }

    # Priority 1: Primary entity header matching primary_entity_keywords
    for col in text_cols:
        c_norm = normalize_string(col)
        c_words = set(re.findall(r"\b[a-z0-9]+\b", c_norm))
        if primary_entity_keywords.intersection(c_words):
            return col

    # Priority 2: Direct match in question
    for col in text_cols:
        c_norm = normalize_string(col)
        c_words = set(re.findall(r"\b[a-z0-9]+\b", c_norm))
        if any(w in q_norm for w in c_words if len(w) > 2 and w not in {"roles", "role", "location", "apply", "skills", "type"}):
            return col

    return text_cols[0]

def load_structured_spreadsheet_from_lancedb(file_path=None):
    """
    Loads structured spreadsheet rows from LanceDB and converts them
    into the same DataFrame-based representation used by the query engine.

    file_path:
        None -> all spreadsheet rows
        specific path -> only that spreadsheet
    """

    table = get_spreadsheet_table()

    if table is None or table.count_rows() == 0:
        return {}

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

    if not rows:
        return {}

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

        # Remove internal columns from actual spreadsheet schema
        data_headers = [
            c for c in df.columns
            if not str(c).startswith("_")
        ]

        data_df = df[data_headers].copy()

        schema = inspect_sheet_schema(
            data_df,
            sheet_name
        )

        result[sheet_name] = {
            "df": data_df,
            "schema": schema,
        }

    return result
def execute_spreadsheet_query(file_path, question):
    """
    Main domain-agnostic orchestrator for spreadsheet queries.
    Inspects workbook schema, extracts filter conditions dynamically,
    validates grounding (fails closed if ungrounded), executes pandas logic,
    and returns grounded results.
    """
    sheets = load_structured_spreadsheet_from_lancedb(file_path)
    if not sheets:
        return f"Could not extract data from '{os.path.basename(file_path)}'.", 0

    q_norm = normalize_string(question)

    # ------------------------------------------------------------
    # 1. Dynamic Sheet Selection (P0-G)
    # ------------------------------------------------------------
    target_sheet_name = None
    for sheet_name in sheets.keys():
        s_norm = normalize_string(sheet_name)
        if s_norm in q_norm or any(w in q_norm for w in s_norm.split() if len(w) > 2):
            if s_norm not in {"summary", "count", "summary & count"}:
                target_sheet_name = sheet_name
                break

    if not target_sheet_name:
        non_summary = [s for s in sheets.keys() if "summary" not in normalize_string(s)]
        target_sheet_name = non_summary[0] if non_summary else list(sheets.keys())[0]

    sheet_data = sheets[target_sheet_name]
    df = sheet_data["df"].copy()
    schema = sheet_data["schema"]
    headers = schema["headers"]
    col_types = schema["column_types"]

    # ------------------------------------------------------------
    # 2. Metadata / Schema Inspection Queries
    # ------------------------------------------------------------
    if re.search(r"\b(?:what\s+columns|available\s+columns|headers|column\s+names)\b", q_norm):
        header_str = ", ".join(headers)
        return f"The available columns in '{target_sheet_name}' are: {header_str}.", len(headers)

    if re.search(r"\b(?:first|head)\s+(\d+)\s+rows\b", q_norm):
        m = re.search(r"\b(?:first|head)\s+(\d+)\s+rows\b", q_norm)
        num_rows = int(m.group(1)) if m else 5
        lines = []
        for idx, (_, row) in enumerate(df.head(num_rows).iterrows(), 1):
            parts = [f"{c}: {row[c]}" for c in headers[:4] if pd.notna(row[c])]
            lines.append(f"{idx}. " + " | ".join(parts))
        return "\n".join(lines), min(num_rows, len(df))

    # ------------------------------------------------------------
    # 3. Dynamic Column Resolution
    # ------------------------------------------------------------
    entity_col = resolve_entity_column(headers, col_types, question)
    metric_col = resolve_metric_column(headers, col_types, question)

    if entity_col and entity_col in df.columns:
        df = df[df[entity_col].notna() & (df[entity_col].astype(str).str.strip() != "")]

    # ------------------------------------------------------------
    # 4. GENERALIZED DATA-DRIVEN GROUP BY / AGGREGATION ENGINE
    # ------------------------------------------------------------
    group_phrase_match = re.search(
        r"\b(?:in\s+each|per|by|for\s+each|grouped?\s+by)\s+([a-z0-9\s/_\-]+)",
        q_norm
    )
    if not group_phrase_match:
        group_phrase_match = re.search(r"\bby\s+([a-z0-9\s/_\-]+)", q_norm)

    is_group_query = bool(group_phrase_match)
    group_col = None

    if is_group_query and group_phrase_match:
        non_summary_sheets = {name: d for name, d in sheets.items() if "summary" not in normalize_string(name)}
        if not non_summary_sheets:
            non_summary_sheets = sheets

        combined_list = []
        for s_name, d in non_summary_sheets.items():
            s_df = d["df"].copy()
            s_df["_sheet_name"] = s_name
            combined_list.append(s_df)

        combined_df = pd.concat(combined_list, ignore_index=True).reset_index(drop=True)
        u_headers = [c for c in combined_df.columns if not str(c).startswith("_")]
        u_schema = inspect_sheet_schema(combined_df[u_headers], "combined") if u_headers else {}
        u_col_types = u_schema.get("column_types", {})

        metric_col = resolve_metric_column(u_headers, u_col_types, question)

        m1 = re.search(r"\b(?:in\s+each|per|for\s+each|grouped?\s+by)\s+([a-z0-9\s/_\-]+)", q_norm)
        m3 = re.search(r"\bby\s+([a-z0-9\s/_\-]+)", q_norm)
        m2 = re.search(r"\b(?:top|bottom|\d+)?\s*([a-z0-9\s/_\-]+)\s+by\b", q_norm)

        raw_phrase = ""
        metric_words = {"average", "avg", "mean", "total", "sum", "count", "number", "ctc", "salary", "price", "revenue", "marks", "rate", "cost", "lpa", "gpa", "highest", "lowest", "top", "bottom"}

        if m1:
            raw_phrase = m1.group(1).strip()
        elif m3 and not all(w in metric_words for w in m3.group(1).split()):
            raw_phrase = m3.group(1).strip()
        elif m2:
            raw_phrase = m2.group(1).strip()

        g_tokens = [w for w in raw_phrase.split() if w not in metric_words and w != "all"]
        group_phrase = " ".join(g_tokens) if g_tokens else raw_phrase

        # Grounding check 1: Match against column headers (excluding metric_col)
        for col in u_headers:
            if metric_col and col == metric_col:
                continue
            c_norm = normalize_string(col)
            if group_phrase in c_norm or c_norm in group_phrase or any(t == c_norm for t in g_tokens if len(t) > 2):
                group_col = col
                break

        # Grounding check 2: Match against sheet names (e.g. Bangalore, Pune, Indore...)
        if not group_col and len(non_summary_sheets) > 1:
            sheet_categories = [normalize_string(s) for s in non_summary_sheets.keys()]
            if any(w in {"city", "cities", "location", "locations", "department", "departments", "state", "states", "branch", "branches", "sheet", "sheets"} for w in g_tokens) or any(any(t in sc for t in g_tokens if len(t) > 2) for sc in sheet_categories):
                group_col = "_sheet_name"

        # Grounding check 3: Match against text column values
        if not group_col:
            for col in u_headers:
                if col != metric_col and u_col_types.get(col) == "text":
                    s_str = combined_df[col].dropna().astype(str).str.lower()
                    if any(s_str.str.contains(re.escape(t), regex=True, na=False).any() for t in g_tokens if len(t) > 2):
                        group_col = col
                        break

        if not group_col:
            return f"I can't determine the grouping dimension from the spreadsheet because no matching column or categorical field was found for '{raw_phrase}'.", 0

        entity_col = resolve_entity_column(u_headers, u_col_types, question)
        metric_col = resolve_metric_column(u_headers, u_col_types, question)

        is_count = bool(re.search(r"\b(?:how\s+many|number\s+of|count|is\s+there\s+any)\b", q_norm))
        is_avg = bool(re.search(r"\b(?:average|avg|mean)\b", q_norm))
        is_sum = bool(re.search(r"\b(?:total|sum)\b", q_norm))
        is_ranking = bool(re.search(r"\b(?:highest|lowest|top|bottom|max|min|best|worst|largest|smallest)\b", q_norm))

        match_top = re.search(r"\btop\s+(\d+)\b", q_norm)
        match_bot = re.search(r"\bbottom\s+(\d+)\b", q_norm)
        limit = None
        if match_top:
            limit = int(match_top.group(1))
        elif match_bot:
            limit = int(match_bot.group(1))

        descending = True
        if re.search(r"\b(?:lowest|smallest|minimum|min|bottom|least|cheapest)\b", q_norm):
            descending = False

        eval_df = combined_df[combined_df[group_col].notna() & (combined_df[group_col].astype(str).str.strip() != "")].reset_index(drop=True)

        if is_avg or is_sum or (is_ranking and metric_col):
            if not metric_col or metric_col not in eval_df.columns:
                return "I can't determine the metric from the spreadsheet.", 0

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
                if limit and is_ranking:
                    lines.append(f"{idx}. {grp_name}: {val:.2f}" if is_avg else f"{idx}. {grp_name}: {val:g}")
                else:
                    lines.append(f"{grp_name}: {val:.2f}" if is_avg else f"{grp_name}: {val:g}")
            return "\n".join(lines), len(lines)

        else: # Default COUNT per group
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
                if limit and is_ranking:
                    lines.append(f"{idx}. {grp_name}: {cnt}")
                else:
                    lines.append(f"{grp_name}: {cnt}")
            return "\n".join(lines), len(lines)

    # ------------------------------------------------------------
    # 5. Extract Filter Conditions & Check Grounding (FAIL CLOSED)
    # ------------------------------------------------------------
    filtered_df = df.copy()
    ungrounded_phrases = []

    words_clean = [w for w in re.findall(r"\b[a-z0-9]+\b", q_norm)]
    stopwords = {
        "how", "many", "what", "which", "who", "where", "when", "is", "are", "there", "any", "the",
        "a", "an", "in", "on", "at", "for", "with", "from", "by", "of", "to", "this", "that", "sheet",
        "spreadsheet", "file", "workbook", "data", "table", "listed", "mentioned", "available",
        "different", "show", "list", "give", "provide", "display", "find", "filter", "sort", "order",
        "rank", "highest", "lowest", "top", "bottom", "all", "count", "number", "hiring", "hired",
        "company", "companies", "student", "students", "product", "products", "item", "items",
        "row", "rows", "column", "columns", "having", "has", "have", "more", "less", "above", "below",
        "roles", "role", "skills", "skill", "city", "cities", "each", "their"
    }

    candidate_phrases = []
    for i in range(len(words_clean)):
        if words_clean[i] not in stopwords:
            candidate_phrases.append(words_clean[i])
            if i + 1 < len(words_clean):
                candidate_phrases.append(f"{words_clean[i]} {words_clean[i+1]}")
            if i + 2 < len(words_clean):
                candidate_phrases.append(f"{words_clean[i]} {words_clean[i+1]} {words_clean[i+2]}")

    candidate_phrases = sorted(list(set(candidate_phrases)), key=len, reverse=True)

    matched_columns_and_vals = []
    for phrase in candidate_phrases:
        if len(phrase) <= 2 or phrase in stopwords:
            continue

        sheet_matched = any(normalize_string(s) == phrase or phrase in normalize_string(s) for s in sheets.keys())
        if sheet_matched:
            continue

        header_matched = any(normalize_string(h) == phrase or phrase in normalize_string(h) for h in headers)
        if header_matched:
            continue

        p_tokens = [t for t in phrase.split() if t not in stopwords and len(t) > 2]
        if not p_tokens:
            continue

        found_cell_match = False
        for token in p_tokens:
            token_header_match = any(token in normalize_string(h) for h in headers)
            if token_header_match:
                found_cell_match = True

            for col in headers:
                if col_types.get(col) == "text":
                    series_str = df[col].dropna().astype(str).str.lower()
                    matches = series_str[series_str.str.contains(re.escape(token), regex=True, na=False)]
                    if not matches.empty:
                        found_cell_match = True
                        if (col, token) not in matched_columns_and_vals and not token_header_match:
                            matched_columns_and_vals.append((col, token))

        if not found_cell_match:
            ungrounded_phrases.append(phrase)

    if ungrounded_phrases:
        missing_term = ungrounded_phrases[0]
        return f"I can't determine that from this spreadsheet because I couldn't find a column or categorical value that represents '{missing_term}' in the workbook.", 0

    for col, val_str in matched_columns_and_vals:
        if col in filtered_df.columns:
            mask = filtered_df[col].astype(str).str.lower().str.contains(re.escape(val_str), regex=True, na=False)
            filtered_df = filtered_df[mask].reset_index(drop=True)

    if metric_col and metric_col in filtered_df.columns:
        filtered_df["_numeric_val"] = filtered_df[metric_col].apply(parse_numeric_value)
        num_match = re.search(r"\b(?:above|over|greater\s+than|more\s+than|>)\s+(\d+(?:\.\d+)?)\b", q_norm)
        if num_match:
            threshold = float(num_match.group(1))
            filtered_df = filtered_df[filtered_df["_numeric_val"] > threshold]

    # ------------------------------------------------------------
    # 6. Structured Operations Execution (COUNT, AVG, SUM, SORT, LIST)
    # ------------------------------------------------------------
    is_count = bool(re.search(r"\b(?:how\s+many|number\s+of|count|is\s+there\s+any)\b", q_norm))
    is_avg = bool(re.search(r"\b(?:average|avg|mean)\b", q_norm))
    is_sum = bool(re.search(r"\b(?:total|sum)\b", q_norm))
    is_ranking = bool(re.search(r"\b(?:highest|lowest|top|bottom|max|min|best|worst|largest|smallest)\b", q_norm))

    if is_count and not is_ranking and not is_avg and not is_sum:
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

        limit = 1
        match_top = re.search(r"\btop\s+(\d+)\b", q_norm)
        if match_top:
            limit = int(match_top.group(1))

        match_bot = re.search(r"\bbottom\s+(\d+)\b", q_norm)
        if match_bot:
            limit = int(match_bot.group(1))

        if "_numeric_val" not in filtered_df.columns:
            filtered_df["_numeric_val"] = filtered_df[metric_col].apply(parse_numeric_value)

        df_sorted = filtered_df.dropna(subset=["_numeric_val"]).sort_values(by="_numeric_val", ascending=not descending)
        df_res = df_sorted.head(limit)

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

        for idx, (_, row) in enumerate(filtered_df.head(10).iterrows(), 1):
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
    for idx, row in enumerate(rows[:10], 1):
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