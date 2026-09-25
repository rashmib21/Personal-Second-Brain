"""
Routing baseline for query_analyzer.analyze_query. Pins what was verified as CORRECT so later edits cannot silently break it.
pytest -q tests/test_routing_baseline.py       env: QA_PATH=<query_analyzer.py>, QP_PATH=<query_plan.py> (only when `app` is not importable)
"""
import importlib.util, logging, os, sys, types
import pytest

QA_PATH = os.environ.get("QA_PATH", "app/query/query_analyzer.py")
QP_PATH = os.environ.get("QP_PATH", "app/query/query_plan.py")
STATE = {}


def _mod(n, **kw):
    m = types.ModuleType(n); m.__path__ = []; m.__dict__.update(kw); sys.modules[n] = m; return m


def _load():
    try:
        from app.query import query_analyzer as qa
        import app.services.interaction_state as istate
        istate.get_last_interaction = lambda: dict(STATE)
        return qa
    except ImportError:
        for n in ["app", "app.utils", "app.storage", "app.query", "app.search", "app.services"]:
            _mod(n)
        _mod("app.utils.logger", logger=logging.getLogger("x"))
        _mod("app.utils.date_parser", parse_date_expression=lambda *a, **k: (None, None, None))
        _mod("app.storage.lancedb_store", get_hash_table=lambda: None)
        _mod("app.search.face_search", is_face_search_query=lambda *a, **k: False,
             extract_person_name_from_question=lambda *a, **k: None)
        _mod("app.services.interaction_state", get_last_interaction=lambda: dict(STATE))
        sp = importlib.util.spec_from_file_location("app.query.query_plan", QP_PATH)
        qp = importlib.util.module_from_spec(sp); sys.modules["app.query.query_plan"] = qp; sp.loader.exec_module(qp)
        sp = importlib.util.spec_from_file_location("qa", QA_PATH)
        qa = importlib.util.module_from_spec(sp); sp.loader.exec_module(qa)
        return qa


qa = _load()
FILES = ["rashmi_photo.jpg", "Rashmi_Barethiya_19-05.pdf", "Cracking the Tech Career.pdf",
         "Germany_IT_Companies_Rashmi_2026.xlsx", "IT_Direct_Hire_Companies_2026.xlsx"]


def run(q):
    r = qa.analyze_query(q, indexed_files=FILES)
    return r.to_dict() if hasattr(r, "to_dict") else r


# (question, intent, source_hint)   source_hint None = "no file pinned, let the workbook picker decide"
CASES = [
    ("can you provide the companies names who are hiring for Data engineer role ??", "SPREADSHEET_QUERY", None),
    ("list of companies hiring for backend roles", "SPREADSHEET_QUERY", None),
    ("how many total companies are listed", "SPREADSHEET_QUERY", None),
    ("how many companies in each city", "SPREADSHEET_QUERY", None),
    # regression: 'IT companies' is a description, NOT the file Germany_IT_Companies_...
    ("show me IT companies in Indore that are hiring for backend developer roles", "SPREADSHEET_QUERY", None),
    ("give me companies hiring for backend roles from IT_Direct_Hire_Companies_2026.xlsx",
     "SPREADSHEET_QUERY", "IT_Direct_Hire_Companies_2026.xlsx"),
    ("list all spreadsheets", "FILE_LIST", None),
    ("tell me the technical skills from Rashmi_Barethiya_19-05.pdf", "TEXT_SEARCH", "Rashmi_Barethiya_19-05.pdf"),
    # regression: file names with spaces must resolve to the real file, not 'career.pdf'
    ("what does Cracking the Tech Career.pdf say about cover letters", "TEXT_SEARCH", "Cracking the Tech Career.pdf"),
    ("show me rashmi_photo.jpg", "IMAGE_VISUAL_QUERY", "rashmi_photo.jpg"),
]


@pytest.mark.parametrize("q,intent,source", CASES, ids=[c[0][:50] for c in CASES])
def test_routing(q, intent, source):
    STATE.clear()
    r = run(q)
    assert r["intent"] == intent
    assert r.get("source_hint") == source


def test_pronoun_followup_keeps_previous_source():
    STATE.update(previous_source="Rashmi_Barethiya_19-05.pdf", retrieved_sources=["Rashmi_Barethiya_19-05.pdf"])
    r = run("tell me the technical skills from it")
    STATE.clear()
    assert r["source_hint"] == "Rashmi_Barethiya_19-05.pdf"
