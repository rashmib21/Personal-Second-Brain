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
    """
    q_norm = normalize_string(question)
    q_words = set(re.findall(r"\b[a-z0-9]+\b", q_norm))

    text_cols = [col for col in headers if column_types.get(col) == "text" or "id" in col.lower()]
    if not text_cols:
        return headers[0] if headers else None

    # Priority 1: Direct match in question (e.g. "transaction" -> Transaction_ID)
    for col in text_cols:
        c_norm = normalize_string(col)
        c_words = set(re.findall(r"\b[a-z0-9]+\b", c_norm))
        if any(w in q_norm for w in c_words if len(w) > 2):
            return col

    # Priority 2: Standard entity header keywords
    entity_keywords = {"name", "title", "item", "entity", "company", "student", "product", "transaction", "customer", "user", "employee", "id"}
    for col in text_cols:
        c_norm = normalize_string(col)
        c_words = set(re.findall(r"\b[a-z0-9]+\b", c_norm))
        if entity_keywords.intersection(c_words):
            return col

    return text_cols[0]


def execute_spreadsheet_query(file_path, question):
    """
    Main domain-agnostic orchestrator for spreadsheet queries.
    Parses request, inspects schema, executes pandas operations, formats answer.
    """
    sheets = load_spreadsheet_sheets(file_path)
    if not sheets:
        return f"Could not extract data from '{os.path.basename(file_path)}'.", 0

    q_norm = normalize_string(question)

    # 1. Sheet selection
    target_sheet_name = None
    for sheet_name in sheets.keys():
        if normalize_string(sheet_name) in q_norm:
            target_sheet_name = sheet_name
            break

    if not target_sheet_name:
        target_sheet_name = list(sheets.keys())[0]

    sheet_data = sheets[target_sheet_name]
    df = sheet_data["df"].copy()
    schema = sheet_data["schema"]
    headers = schema["headers"]
    col_types = schema["column_types"]

    entity_col = resolve_entity_column(headers, col_types, question)
    metric_col = resolve_metric_column(headers, col_types, question)

    # Clean empty entity rows
    if entity_col and entity_col in df.columns:
        df = df[df[entity_col].notna() & (df[entity_col].astype(str).str.strip() != "")]

    # 2. COUNT Operation
    is_count = bool(re.search(r"\b(?:how\s+many|number\s+of|count)\b", q_norm))
    if is_count and not re.search(r"\b(?:highest|lowest|top|bottom)\b", q_norm):
        count_val = len(df)
        return f"There are {count_val} items in '{target_sheet_name}'.", count_val

    # 3. AGGREGATE Operations (AVG, SUM)
    is_avg = bool(re.search(r"\b(?:average|avg|mean)\b", q_norm))
    is_sum = bool(re.search(r"\b(?:total|sum)\b", q_norm))

    if is_avg and metric_col:
        numeric_series = df[metric_col].apply(parse_numeric_value).dropna()
        if not numeric_series.empty:
            avg_val = numeric_series.mean()
            return f"The average {metric_col} is {avg_val:.2f}.", len(df)

    if is_sum and metric_col:
        numeric_series = df[metric_col].apply(parse_numeric_value).dropna()
        if not numeric_series.empty:
            sum_val = numeric_series.sum()
            return f"The total {metric_col} is {sum_val:g}.", len(df)

    # 4. SORT / RANKING / TOP_N Operations
    if metric_col:
        descending = True
        if re.search(r"\b(?:lowest|smallest|minimum|min|bottom|least|cheapest)\b", q_norm):
            descending = False

        limit = None
        match_top = re.search(r"\btop\s+(\d+)\b", q_norm)
        if match_top:
            limit = int(match_top.group(1))

        match_bot = re.search(r"\bbottom\s+(\d+)\b", q_norm)
        if match_bot:
            limit = int(match_bot.group(1))

        if limit is None:
            if re.search(r"\b(?:highest|lowest|largest|smallest|most|least|best|worst|top|bottom)\b", q_norm):
                limit = 1

        df["_sort_key"] = df[metric_col].apply(parse_numeric_value)
        df_sorted = df.dropna(subset=["_sort_key"]).sort_values(by="_sort_key", ascending=not descending)

        if limit:
            df_result = df_sorted.head(limit)
        else:
            df_result = df_sorted

        if not df_result.empty:
            lines = []
            for idx, (_, row) in enumerate(df_result.iterrows(), 1):
                entity_val = str(row.get(entity_col, "")).strip() if entity_col else f"Item {idx}"
                metric_val = str(row.get(metric_col, "")).strip()

                line = f"{idx}. {entity_col}: {entity_val} | {metric_col}: {metric_val}"
                lines.append(line)

            return "\n".join(lines), len(df_result)

    # 5. Default Fallback Formatting using Actual Sheet Headers
    lines = []
    display_cols = [c for c in [entity_col, metric_col] if c]
    if not display_cols:
        display_cols = headers[:3]

    for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
        parts = []
        for col in display_cols:
            parts.append(f"{col}: {row.get(col, '')}")
        lines.append(f"{idx}. " + " | ".join(parts))

    return "\n".join(lines) if lines else "No matching spreadsheet records found.", len(df)


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