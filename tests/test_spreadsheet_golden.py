"""
Golden + oracle + metamorphic tests for the spreadsheet engine.
Run:  pytest -q tests/test_spreadsheet_golden.py
Env:  TEST_XLSX=<path to IT_Direct_Hire_Companies_2026.xlsx>   (default: watched_folder/...)
      SQ_PATH=<path to spreadsheet_query.py> (only used when `app` package is not importable)

Tests run in NO-LLM mode: the query planner is forced to return None, so we test the
deterministic layer only. A change is safe only if this file stays green.
"""
import importlib.util, os, re, sys, types
import pandas as pd
import pytest

XLSX = os.environ.get("TEST_XLSX", "watched_folder/IT_Direct_Hire_Companies_2026.xlsx")
SQ_PATH = os.environ.get("SQ_PATH", "app/rag/spreadsheet_query.py")


def _load_engine():
    try:
        import app.rag.spreadsheet_query as m
        return m
    except ImportError:
        for name in ["app", "app.storage", "app.storage.lancedb_store", "app.llm", "app.llm.ollama_client",
                     "app.services", "app.services.interaction_state"]:
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["app.storage.lancedb_store"].get_spreadsheet_table = lambda: None
        sys.modules["app.llm.ollama_client"].ask_llama = lambda *a, **k: ""
        sys.modules["app.llm.ollama_client"].ask_spreadsheet_llm = lambda *a, **k: ""
        st = sys.modules["app.services.interaction_state"]
        st.get_last_interaction = lambda: {}
        st.set_spreadsheet_context = lambda *a, **k: None
        st.get_spreadsheet_context = lambda: {}
        spec = importlib.util.spec_from_file_location("sq", SQ_PATH)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


sq = _load_engine()


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setattr(sq, "generate_structured_query_plan", lambda *a, **k: None)
    monkeypatch.setattr(sq, "get_last_interaction", lambda: {}, raising=False)
    monkeypatch.setattr(sq, "get_spreadsheet_context", lambda: {}, raising=False)


# ---------- independent oracle (plain pandas, does NOT use the engine) ----------
def oracle() -> pd.DataFrame:
    frames = []
    for sheet, raw in pd.read_excel(XLSX, sheet_name=None, header=None, dtype=object).items():
        hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Type"]
        if len(hdr) == 0:
            continue                                   # summary sheet
        h = hdr[0]
        df = raw.iloc[h + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[h]]
        df = df[df["Company Name"].notna()]            # drops legend + '── SECTION ──' rows
        df["City"] = sheet.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


ORA = oracle()
ROLES = "Roles (Fresher)"


def has_word(series, pattern):
    return series.astype(str).str.contains(pattern, case=False, regex=True, na=False)


def names(answer: str):
    return {m.group(1).strip() for m in re.finditer(r"Company Name: (.*?)(?: \||$)", answer, re.M)}


def ask(q):
    return sq.execute_spreadsheet_query(XLSX, q)


# ---------- 1. data-loading invariants ----------
def test_loader_reads_every_data_row():
    sheets = sq.load_spreadsheet_sheets(XLSX)
    assert sum(len(d["df"]) for d in sheets.values()) == len(ORA)
    assert set(sheets) == set(ORA["City"])


def test_loader_uses_real_header_not_banner():
    for d in sq.load_spreadsheet_sheets(XLSX).values():
        assert "Company Name" in d["schema"]["headers"]


# ---------- 2. oracle tests: engine result set == pandas result set ----------
BACKEND = r"\bbackend\b"
CASES = [
    ("list of companies hiring for backend roles", ORA[has_word(ORA[ROLES], BACKEND)]),
    ("show me IT companies in Indore that are hiring for backend developer roles",
     ORA[(ORA.City == "Indore") & has_word(ORA[ROLES], BACKEND)]),
    ("companies in Pune hiring for backend roles", ORA[(ORA.City == "Pune") & has_word(ORA[ROLES], BACKEND)]),
    ("companies with python skills in Ahmedabad",
     ORA[(ORA.City == "Ahmedabad") & has_word(ORA["Key Skills"], r"\bpython\b")]),
]


@pytest.mark.parametrize("q,expected", CASES, ids=[c[0][:45] for c in CASES])
def test_result_set_matches_oracle(q, expected):
    answer, n = ask(q)
    assert n == len(expected), f"{q!r}: engine={n} oracle={len(expected)}"
    assert names(answer) == set(expected["Company Name"].astype(str).str.strip())


# ---------- 3. metamorphic tests: same meaning => same result ----------
PARAPHRASES = [
    "list of companies hiring for backend roles",
    "Show me all companies which are hiring for BACKEND roles.",
    "give me companies hiring for backend developer roles",
    "give me companies hiring for backend developers",      # plural
    "backend roles companies list",                         # word order
]


def test_paraphrases_give_same_answer():
    base = names(ask(PARAPHRASES[0])[0])
    assert len(base) > 0
    # 'developer(s)' may legitimately narrow to rows that also say Dev/Developer
    for q in PARAPHRASES[:2] + PARAPHRASES[4:]:
        assert names(ask(q)[0]) == base, q


def test_filename_in_question_is_ignored():
    a = names(ask("list of companies hiring for backend roles")[0])
    b = names(ask("list of companies hiring for backend roles in IT_Direct_Hire_Companies_2026.xlsx")[0])
    assert a == b


def test_plural_singular_equivalent():
    assert ask("backend developers in Indore")[1] == ask("backend developer in Indore")[1]


# ---------- 4. counts / aggregates ----------
def test_count_with_city():
    assert ask("how many companies hiring for backend in Pune")[1] == len(
        ORA[(ORA.City == "Pune") & has_word(ORA[ROLES], BACKEND)])


def test_total_rows_and_unique_names():
    answer, n = ask("how many total companies are listed in IT_Direct_Hire_Companies_2026.xlsx")
    assert n == ORA["Company Name"].nunique()
    assert str(len(ORA)) in answer          # engine must also disclose the row count


def test_group_by_city():
    answer, _ = ask("how many companies in each city")
    for city, cnt in ORA.groupby("City")["Company Name"].nunique().items():
        assert re.search(rf"{city}: {cnt}\b", answer), (city, cnt, answer)


# ---------- 5. fail-closed must still work ----------
@pytest.mark.parametrize("q", [
    "companies in Atlantis hiring for backend roles",
    "companies that use cobol mainframes",
])
def test_unknown_term_fails_closed(q):
    answer, n = ask(q)
    assert n == 0 and "can't determine" in answer.lower()
