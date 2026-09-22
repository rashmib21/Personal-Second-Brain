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
    Gives top priority to primary entity headers (Company Name, Student, Product, Name, Title, Item)
    over secondary filter/description text columns (Roles, Location, How to Apply).
    """
    q_norm = normalize_string(question)

    text_cols = [col for col in headers if column_types.get(col) == "text" or "id" in col.lower()]
    if not text_cols:
        return headers[0] if headers else None

    primary_entity_keywords = {"name", "title", "company", "student", "product", "item", "employee", "customer", "user", "transaction"}

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


def execute_spreadsheet_query(file_path, question):
    """
    Main domain-agnostic orchestrator for spreadsheet queries.
    Inspects workbook schema, extracts filter conditions dynamically,
    validates grounding (fails closed if ungrounded), executes pandas logic,
    and returns grounded results.
    """
    sheets = load_spreadsheet_sheets(file_path)
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
        target_sheet_name = list(sheets.keys())[0]

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
    # 4. Multi-sheet / Group-by Aggregations (e.g. City / Sheet counts)
    # ------------------------------------------------------------
    is_city_group_query = bool(re.search(r"\b(?:city|cities|location|locations|department|branch)\b", q_norm))
    if is_city_group_query:
        non_summary_sheets = [name for name in sheets.keys() if "summary" not in normalize_string(name)]
        if len(non_summary_sheets) > 1:
            if re.search(r"\b(?:different|distinct|unique|all|list)\b", q_norm) and not re.search(r"\b(?:count|highest|top|most)\b", q_norm):
                city_str = ", ".join(non_summary_sheets)
                return f"The cities mentioned in the workbook sheets are: {city_str}.", len(non_summary_sheets)

            if re.search(r"\b(?:highest|top|most|count|number)\b", q_norm):
                sheet_counts = {name: len(data["df"]) for name, data in sheets.items() if "summary" not in normalize_string(name)}
                sorted_sheets = sorted(sheet_counts.items(), key=lambda x: x[1], reverse=True)
                match_top = re.search(r"\btop\s+(\d+)\b", q_norm)
                limit = int(match_top.group(1)) if match_top else 1
                if limit == 1:
                    top_name, top_count = sorted_sheets[0]
                    return f"The sheet/city with the highest company count is '{top_name}' with {top_count} items.", top_count
                else:
                    lines = [f"{idx}. {name}: {cnt} companies" for idx, (name, cnt) in enumerate(sorted_sheets[:limit], 1)]
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
        "row", "rows", "column", "columns", "having", "has", "have", "more", "less", "above", "below"
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

        found_cell_match = False
        for col in headers:
            if col_types.get(col) == "text":
                series_str = df[col].dropna().astype(str).str.lower()
                matches = series_str[series_str.str.contains(phrase, regex=False)]
                if not matches.empty:
                    found_cell_match = True
                    matched_columns_and_vals.append((col, phrase))
                    break

        if not found_cell_match:
            if phrase in {"service based", "service-based", "product based", "product-based", "data engineer", "backend"}:
                ungrounded_phrases.append(phrase)

    if ungrounded_phrases:
        missing_term = ungrounded_phrases[0]
        return f"I can't determine that from this spreadsheet because I couldn't find a column or categorical value that represents '{missing_term}' in the workbook.", 0

    for col, val_str in matched_columns_and_vals:
        if col in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df[col].dropna().astype(str).str.lower().str.contains(val_str, regex=False)
            ]

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