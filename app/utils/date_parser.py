import re
import calendar
from datetime import datetime, timedelta

def parse_date_expression(text_input, now=None):
    """
    Fully generic dynamic date parser.
    Normalizes natural-language relative date expressions and absolute date formats
    into standard internal (start_datetime, end_datetime, display_label) tuples relative to 'now' (defaulting to datetime.now()).
    
    Supports:
    - Relative: 'today', 'yesterday', 'day before yesterday', 'N days ago', 'this week', 'last week', 'this month', 'last month', 'recently'/'latest'
    - Absolute: '2026-09-10', '10/09/2026', '10-09-2026', '10 September 2026', '10 Sep 2026', 'September 10 2026'
    
    Returns tuple: (start_datetime, end_datetime, display_label)
    If no date expression is detected, returns (None, None, "all time").
    """
    if not text_input:
        return None, None, "all time"

    if now is None:
        now = datetime.now()

    text_lower = text_input.lower().strip()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1) - timedelta(microseconds=1)

    # 1. Relative Expression: 'today'
    if re.search(r"\btoday\b", text_lower):
        return today_start, today_end, "today (" + today_start.strftime("%B %d, %Y") + ")"

    # 2. Relative Expression: 'day before yesterday'
    if re.search(r"\bday\s+before\s+yesterday\b", text_lower):
        d_start = today_start - timedelta(days=2)
        d_end = d_start + timedelta(days=1) - timedelta(microseconds=1)
        return d_start, d_end, "day before yesterday (" + d_start.strftime("%B %d, %Y") + ")"

    # 3. Relative Expression: 'yesterday'
    if re.search(r"\byesterday\b", text_lower):
        y_start = today_start - timedelta(days=1)
        y_end = today_start - timedelta(microseconds=1)
        return y_start, y_end, "yesterday (" + y_start.strftime("%B %d, %Y") + ")"

    # 4. Relative Expression: 'N days ago'
    days_ago_match = re.search(r"\b(\d+)\s+days?\s+ago\b", text_lower)
    if days_ago_match:
        n_days = int(days_ago_match.group(1))
        n_start = today_start - timedelta(days=n_days)
        n_end = n_start + timedelta(days=1) - timedelta(microseconds=1)
        return n_start, n_end, f"{n_days} days ago (" + n_start.strftime("%B %d, %Y") + ")"

    # 5. Relative Expression: 'this week'
    if re.search(r"\bthis\s+week\b", text_lower):
        w_start = today_start - timedelta(days=today_start.weekday())
        w_end = w_start + timedelta(days=7) - timedelta(microseconds=1)
        return w_start, w_end, "this week (" + w_start.strftime("%b %d") + " - " + w_end.strftime("%b %d, %Y") + ")"

    # 6. Relative Expression: 'last week'
    if re.search(r"\blast\s+week\b", text_lower):
        lw_start = today_start - timedelta(days=today_start.weekday() + 7)
        lw_end = lw_start + timedelta(days=7) - timedelta(microseconds=1)
        return lw_start, lw_end, "last week (" + lw_start.strftime("%b %d") + " - " + lw_end.strftime("%b %d, %Y") + ")"

    # 7. Relative Expression: 'this month'
    if re.search(r"\bthis\s+month\b", text_lower):
        m_start = today_start.replace(day=1)
        next_month = m_start.replace(day=28) + timedelta(days=4)
        m_end = next_month.replace(day=1) - timedelta(microseconds=1)
        return m_start, m_end, "this month (" + m_start.strftime("%B %Y") + ")"

    # 8. Relative Expression: 'last month'
    if re.search(r"\blast\s+month\b", text_lower):
        m_start = (today_start.replace(day=1) - timedelta(days=1)).replace(day=1)
        next_month = m_start.replace(day=28) + timedelta(days=4)
        m_end = next_month.replace(day=1) - timedelta(microseconds=1)
        return m_start, m_end, "last month (" + m_start.strftime("%B %Y") + ")"

    # 9. Absolute Date: YYYY-MM-DD
    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text_lower)
    if iso_match:
        try:
            yr, mo, dy = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            dt_start = datetime(yr, mo, dy, 0, 0, 0)
            dt_end = dt_start + timedelta(days=1) - timedelta(microseconds=1)
            return dt_start, dt_end, dt_start.strftime("%B %d, %Y")
        except ValueError:
            pass

    # 10. Absolute Date: Named Month (e.g. '10 September 2026', '10 Sep 2026', 'September 10 2026')
    month_names = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12
    }

    month_pattern = r"\b(" + "|".join(month_names.keys()) + r")\b"
    m_match = re.search(month_pattern, text_lower)
    if m_match:
        m_str = m_match.group(1)
        m_num = month_names[m_str]

        # Look for day number nearby
        day_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text_lower)
        year_match = re.search(r"\b(20\d{2})\b", text_lower)

        yr = int(year_match.group(1)) if year_match else now.year
        dy = int(day_match.group(1)) if day_match and int(day_match.group(1)) <= 31 else 1

        try:
            dt_start = datetime(yr, m_num, dy, 0, 0, 0)
            dt_end = dt_start + timedelta(days=1) - timedelta(microseconds=1)
            return dt_start, dt_end, dt_start.strftime("%B %d, %Y")
        except ValueError:
            pass

    # 11. Absolute Date: DD/MM/YYYY or DD-MM-YYYY or MM/DD/YYYY
    slash_match = re.search(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b", text_lower)
    if slash_match:
        part1, part2, yr_part = int(slash_match.group(1)), int(slash_match.group(2)), int(slash_match.group(3))
        yr = yr_part if yr_part > 100 else 2000 + yr_part

        # Try DD/MM/YYYY first
        try:
            dt_start = datetime(yr, part2, part1, 0, 0, 0)
            dt_end = dt_start + timedelta(days=1) - timedelta(microseconds=1)
            return dt_start, dt_end, dt_start.strftime("%B %d, %Y")
        except ValueError:
            try:
                # Try MM/DD/YYYY
                dt_start = datetime(yr, part1, part2, 0, 0, 0)
                dt_end = dt_start + timedelta(days=1) - timedelta(microseconds=1)
                return dt_start, dt_end, dt_start.strftime("%B %d, %Y")
            except ValueError:
                pass

    return None, None, "all time"
