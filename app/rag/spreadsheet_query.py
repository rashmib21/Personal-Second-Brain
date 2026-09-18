import csv
import os
import re

from openpyxl import load_workbook


# Only these sheets contain company records in the current workbook.
# Summary/count sheets are intentionally excluded from structured queries.
SUMMARY_SHEET_NAMES = {
    "summary & count",
}


def _normalize_header(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower()
    )


def _find_column(headers, *names):
    normalized = {
        _normalize_header(header): header
        for header in headers
        if header is not None
    }

    for name in names:
        key = _normalize_header(name)

        if key in normalized:
            return normalized[key]

    return None


def _is_real_company_row(row, company_column):
    if not company_column:
        return False

    value = row.get(company_column)

    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    # Ignore section labels such as:
    # ── PRODUCT ──
    # ── GCC/GLOBAL ──
    if text.startswith("──") or text.endswith("──"):
        return False

    return True


def _read_xlsx(path):
    workbook = load_workbook(
        path,
        data_only=True
    )

    rows = []

    for sheet in workbook.worksheets:

        sheet_name = sheet.title.strip()

        # Ignore summary/count sheet.
        if sheet_name.lower() in SUMMARY_SHEET_NAMES:
            continue

        values = list(
            sheet.iter_rows(
                values_only=True
            )
        )

        if not values:
            continue

        # Find the actual header row instead of assuming row 1.
        header_index = None

        for index, row in enumerate(values):
            normalized_values = {
                _normalize_header(value)
                for value in row
                if value is not None
            }

            if (
                "type" in normalized_values
                and "company name" in normalized_values
                and "ctc (lpa)" in normalized_values
            ):
                header_index = index
                break

        if header_index is None:
            continue

        headers = list(values[header_index])

        company_column = _find_column(
            headers,
            "Company Name"
        )

        if not company_column:
            continue

        for row in values[header_index + 1:]:

            record = {}

            for index, header in enumerate(headers):
                if header is None:
                    continue

                value = (
                    row[index]
                    if index < len(row)
                    else None
                )

                record[str(header).strip()] = value

            if not _is_real_company_row(
                record,
                company_column
            ):
                continue

            # Preserve city/sheet information.
            record["_sheet"] = sheet_name

            rows.append(record)

    return rows


def _read_csv(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if any(
                value is not None
                and str(value).strip()
                for value in row.values()
            ):
                rows.append(dict(row))

    return rows


def load_spreadsheet_rows(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".xlsx":
        return _read_xlsx(path)

    if extension == ".csv":
        return _read_csv(path)

    raise ValueError(
        "Structured spreadsheet queries currently "
        f"support .xlsx and .csv files, not {extension}"
    )


def _contains(value, text):
    if value is None:
        return False

    return text.lower() in str(value).lower()


def _get_headers(rows):
    if not rows:
        return []

    return list(rows[0].keys())


def _validate_requested_numeric_field(rows, question):
    """
    Validate that a requested numeric field actually exists.

    Important:
    - salary must resolve to a real salary column
    - ctc must resolve to a real CTC column
    - package must resolve to a real package column
    - lpa may resolve to an LPA/CTC(LPA) column

    Never silently substitute CTC for salary.
    """
    if not rows:
        return None

    headers = _get_headers(rows)
    question_lower = question.lower()

    requested = None

    if re.search(r"\bsalary\b", question_lower):
        requested = (
            "salary",
            ("Salary", "Monthly Salary", "Annual Salary",
             "Base Salary", "Salary (LPA)", "Annual Salary (LPA)")
        )
    elif re.search(r"\bctc\b", question_lower):
        requested = (
            "CTC",
            ("CTC (LPA)", "CTC")
        )
    elif re.search(r"\bpackage\b", question_lower):
        requested = (
            "package",
            ("Package", "Package (LPA)")
        )
    elif re.search(r"\blpa\b", question_lower):
        requested = (
            "LPA",
            ("LPA", "Salary (LPA)", "CTC (LPA)")
        )

    if requested is None:
        return None

    field_name, candidates = requested
    column = _find_column(headers, *candidates)

    if column is None:
        return {
            "field": field_name,
            "column": None,
            "available_columns": headers,
        }

    return {
        "field": field_name,
        "column": column,
        "available_columns": headers,
    }


def filter_rows(rows, question):
    if not rows:
        return []

    question_lower = question.lower()

    headers = _get_headers(rows)

    filtered = list(rows)

    # ---------------------------------------------------------
    # PRODUCT
    #
    # "product based" means Type == Product.
    #
    # Do NOT include Analytics or GCC/Global.
    # ---------------------------------------------------------
    if re.search(
        r"\bproduct(?:[- ]based)?\b",
        question_lower
    ):
        type_column = _find_column(
            headers,
            "Type"
        )

        if type_column:
            filtered = [
                row
                for row in filtered
                if str(
                    row.get(type_column, "")
                ).strip().lower() == "product"
            ]

    # ---------------------------------------------------------
    # BACKEND
    # ---------------------------------------------------------
    if re.search(
        r"\bbackend\b",
        question_lower
    ):
        role_column = _find_column(
            headers,
            "Roles (Fresher)",
            "Roles"
        )

        if role_column:
            filtered = [
                row
                for row in filtered
                if _contains(
                    row.get(role_column),
                    "backend"
                )
            ]

    # ---------------------------------------------------------
    # FRONTEND
    # ---------------------------------------------------------
    if re.search(
        r"\bfront[\s-]?end\b",
        question_lower
    ):
        role_column = _find_column(
            headers,
            "Roles (Fresher)",
            "Roles"
        )

        if role_column:
            filtered = [
                row
                for row in filtered
                if _contains(
                    row.get(role_column),
                    "frontend"
                )
            ]

    # ---------------------------------------------------------
    # FULL STACK
    # ---------------------------------------------------------
    if re.search(
        r"\bfull[\s-]?stack\b",
        question_lower
    ):
        role_column = _find_column(
            headers,
            "Roles (Fresher)",
            "Roles"
        )

        if role_column:
            filtered = [
                row
                for row in filtered
                if _contains(
                    row.get(role_column),
                    "full stack"
                )
            ]

    # ---------------------------------------------------------
    # CITY
    #
    # The workbook uses sheet names as cities.
    # ---------------------------------------------------------
    city_names = {
        "bangalore",
        "bengaluru",
        "pune",
        "hyderabad",
        "indore",
        "ahmedabad",
    }

    requested_city = None

    for city in city_names:
        if re.search(
            rf"\b{re.escape(city)}\b",
            question_lower
        ):
            requested_city = city
            break

    if requested_city:
        city_aliases = {
            "bengaluru": "bangalore",
        }

        requested_city = city_aliases.get(
            requested_city,
            requested_city
        )

        filtered = [
            row
            for row in filtered
            if str(
                row.get("_sheet", "")
            ).strip().lower() == requested_city
        ]

    # ---------------------------------------------------------
    # AREA / LOCATION
    #
    # Detect known location phrases after "in", "near", etc.
    # without accidentally treating "in the folder" as a
    # company location.
    # ---------------------------------------------------------
    location_column = _find_column(
        headers,
        "Area / Location",
        "Location",
        "Area"
    )

    known_locations = [
        "koramangala",
        "indiranagar",
        "whitefield",
        "electronic city",
        "hsr layout",
        "marathahalli",
        "kharadi",
        "hinjewadi",
        "magarpatta",
        "baner",
        "wakad",
        "gachibowli",
        "hitech city",
        "madhapur",
        "scheme 94",
        "scheme 78",
        "vijay nagar",
        "new palasia",
        "palasia",
        "prahladnagar",
        "bodakdev",
        "satellite",
    ]

    requested_location = None

    for location in known_locations:
        if re.search(
            rf"\b{re.escape(location)}\b",
            question_lower
        ):
            requested_location = location
            break

    if requested_location and location_column:
        filtered = [
            row
            for row in filtered
            if _contains(
                row.get(location_column),
                requested_location
            )
        ]

    return filtered


def _ctc_sort_key(value):
    """
    Converts CTC values such as:

        8–15
        8-15
        3–5.5
        18–30

    into a sortable tuple:

        (minimum, maximum)
    """

    if value is None:
        return (
            float("inf"),
            float("inf")
        )

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        str(value)
    )

    if not numbers:
        return (
            float("inf"),
            float("inf")
        )

    values = [
        float(number)
        for number in numbers
    ]

    return (
        values[0],
        values[-1]
    )


def _find_numeric_column(headers, *names):
    """
    Find a numeric compensation/metric column only when the
    requested semantic field actually exists in the spreadsheet.

    This deliberately does NOT treat salary and CTC as synonyms.
    """
    return _find_column(headers, *names)


def _requested_numeric_field(headers, question):
    """
    Resolve the numeric field requested by the user.

    Important:
    - salary -> Salary / Monthly Salary / Annual Salary
    - ctc -> CTC
    - package -> Package
    - lpa -> LPA/CTC-style fields

    Salary must never silently fall back to CTC.
    """
    question_lower = question.lower()

    if re.search(r"\bsalary\b", question_lower):
        return _find_numeric_column(
            headers,
            "Salary",
            "Monthly Salary",
            "Annual Salary",
            "Base Salary",
            "Salary (LPA)",
            "Annual Salary (LPA)"
        )

    if re.search(r"\bctc\b", question_lower):
        return _find_numeric_column(
            headers,
            "CTC (LPA)",
            "CTC"
        )

    if re.search(r"\bpackage\b", question_lower):
        return _find_numeric_column(
            headers,
            "Package",
            "Package (LPA)"
        )

    if re.search(r"\blpa\b", question_lower):
        return _find_numeric_column(
            headers,
            "LPA",
            "Salary (LPA)",
            "CTC (LPA)"
        )

    return None


def is_single_ranking_query(question):
    """
    Return True when the user asks for one best/worst row,
    rather than asking to sort the entire spreadsheet.
    """
    q = question.lower()

    directional_sort = bool(
        re.search(
            r"\b(?:highest|lowest|largest|smallest)\s+to\s+"
            r"(?:lowest|highest|largest|smallest)\b",
            q
        )
        or re.search(r"\b(?:ascending|descending)\b", q)
    )

    explicit_top_n = bool(
        re.search(r"\btop\s+\d+\b", q)
        or re.search(r"\bbottom\s+\d+\b", q)
    )

    single_rank = bool(
        re.search(
            r"\b(?:highest|lowest|largest|smallest|maximum|minimum|max|min)\b",
            q
        )
    )

    return single_rank and not directional_sort and not explicit_top_n


def requested_result_limit(question):
    """
    Return the number of rows requested by a ranking query.

    Examples:
      highest CTC -> 1
      lowest salary -> 1
      top 5 CTC -> 5
      bottom 3 CTC -> 3
      normal sort -> None
    """
    q = question.lower()

    match = re.search(r"\btop\s+(\d+)\b", q)
    if match:
        return int(match.group(1))

    match = re.search(r"\bbottom\s+(\d+)\b", q)
    if match:
        return int(match.group(1))

    if is_single_ranking_query(q):
        return 1

    return None


def format_ranking_result(rows, question):
    """
    Format ranking results using only the columns relevant
    to the user's question.

    Example:
      Which company has the highest CTC?
      -> Postman | CTC: 10–20
    """
    if not rows:
        return "No matching rows were found."

    headers = _get_headers(rows)
    q = question.lower()

    # Resolve the requested numeric field.
    requested = None

    if re.search(r"\bsalary\b", q):
        requested = _find_column(
            headers,
            "Salary",
            "Monthly Salary",
            "Annual Salary",
            "Base Salary",
            "Salary (LPA)",
            "Annual Salary (LPA)"
        )
    elif re.search(r"\bctc\b", q):
        requested = _find_column(
            headers,
            "CTC (LPA)",
            "CTC"
        )
    elif re.search(r"\bpackage\b", q):
        requested = _find_column(
            headers,
            "Package",
            "Package (LPA)"
        )
    elif re.search(r"\blpa\b", q):
        requested = _find_column(
            headers,
            "LPA",
            "Salary (LPA)",
            "CTC (LPA)"
        )

    # Resolve the entity/name column generically.
    name_column = _find_column(
        headers,
        "Company Name",
        "Company",
        "Employee Name",
        "Employee",
        "Name",
        "Person Name"
    )

    if requested and name_column:
        return (
            f"{name_column}: {rows[0].get(name_column, '')} | "
            f"{requested}: {rows[0].get(requested, '')}"
        )

    # Fallback: return the first row without dumping the entire sheet.
    return format_rows(rows[:1])


def sort_rows(rows, question):
    if not rows:
        return []

    question_lower = question.lower()

    is_sort_query = bool(
        re.search(
            r"\b(?:sort|sorted|order|rank)\b",
            question_lower
        )
    )

    if not is_sort_query:
        return rows

    headers = _get_headers(rows)

    requested_column = _requested_numeric_field(
        headers,
        question
    )

    if requested_column:
        descending = bool(
            re.search(
                r"\b(?:highest|descending|largest|max|maximum|highest to lowest)\b",
                question_lower
            )
        )

        return sorted(
            rows,
            key=lambda row: _ctc_sort_key(
                row.get(requested_column)
            ),
            reverse=descending
        )

    return rows


def group_by_city(rows):
    groups = {}

    for row in rows:
        city = str(
            row.get("_sheet", "Unknown")
        ).strip()

        groups.setdefault(
            city,
            []
        ).append(row)

    return groups


def format_rows(rows):
    if not rows:
        return "No matching companies found."

    headers = _get_headers(rows)

    company_column = _find_column(
        headers,
        "Company Name"
    )

    type_column = _find_column(
        headers,
        "Type"
    )

    role_column = _find_column(
        headers,
        "Roles (Fresher)",
        "Roles"
    )

    ctc_column = _find_column(
        headers,
        "CTC (LPA)",
        "CTC",
        "Package"
    )

    location_column = _find_column(
        headers,
        "Area / Location",
        "Location",
        "Area"
    )

    lines = []

    for index, row in enumerate(rows, 1):

        company = (
            row.get(company_column, "Unknown")
            if company_column
            else "Unknown"
        )

        company_type = (
            row.get(type_column, "")
            if type_column
            else ""
        )

        roles = (
            row.get(role_column, "")
            if role_column
            else ""
        )

        ctc = (
            row.get(ctc_column, "")
            if ctc_column
            else ""
        )

        location = (
            row.get(location_column, "")
            if location_column
            else ""
        )

        city = row.get(
            "_sheet",
            ""
        )

        parts = [
            f"{index}. {company}"
        ]

        if city:
            parts.append(
                f"City: {city}"
            )

        if company_type:
            parts.append(
                f"Type: {company_type}"
            )

        if roles:
            parts.append(
                f"Roles: {roles}"
            )

        if ctc:
            parts.append(
                f"CTC: {ctc}"
            )

        if location:
            parts.append(
                f"Location: {location}"
            )

        lines.append(
            " | ".join(parts)
        )

    return "\n".join(lines)   