"""pytest -q tests/test_conversation.py   (imports the modules from conversation/ or app/query/)"""
import importlib.util, os, sys, pytest

BASE = os.environ.get("CONV_DIR", os.path.join(os.path.dirname(__file__), "..", "app", "query"))


def _load(name):
    if name == "interaction_state":
        try:
            import app.services.interaction_state as m
            return m
        except ImportError:
            pass
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), "..", "app", "services", name + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cl = _load("clarification")
sc = _load("source_catalog")
st = _load("interaction_state")

CANDS = ["rashmi_photo.jpg", "Rashmi_Barethiya_19-05.pdf"]
PDF, JPG = CANDS[1], CANDS[0]


# ---- 1. 'Which one did you mean?' replies -> the original question must resume ----
@pytest.mark.parametrize("reply,expected", [
    ("Rashmi_Barethiya_19-05.pdf.", PDF),          # the exact reply from the real log
    ("Rashmi_Barethiya_19-05", PDF),
    ("rashmi_barethiya_19-05.pdf", PDF),
    ("Rashmi_Barethya_19-05.pdf", PDF),            # typo
    ("the pdf", PDF), ("pdf wala", PDF), ("the document one", PDF),
    ("photo", JPG), ("the image one", JPG), ("jpg", JPG), ("rashmi_photo", JPG),
    ("first one", JPG), ("pehla wala", JPG), ("2", PDF), ("the second", PDF), ("dusra", PDF), ("last", PDF),
    ("barethiya", PDF),
])
def test_selection_replies(reply, expected):
    assert cl.resolve_selection(reply, CANDS) == expected


@pytest.mark.parametrize("reply", [
    "tell me the technical skills from the resume.",     # new question, must NOT be a pick
    "what is docker", "rashmi",                          # 'rashmi' matches both files -> ambiguous
    "yes", "", "the pdf and the photo", "5",
])
def test_non_selection_replies(reply):
    assert cl.resolve_selection(reply, CANDS) is None


# ---- 2. pending state round-trip ----
def test_pending_clarification_roundtrip():
    st.clear_last_interaction()
    st.set_pending_clarification("tell me the technical skills of rashmi?", CANDS)
    p = st.get_pending_clarification()
    assert p["query"].startswith("tell me") and p["candidates"] == CANDS
    st.clear_pending_clarification()
    assert st.get_pending_clarification() == {}


# ---- 3. follow-ups without a named source ----
STATE = {"last_document_source": PDF, "last_image_source": JPG, "previous_source": PDF}


@pytest.mark.parametrize("q,expected", [
    ("what are his projects there", PDF), ("and its education?", PDF), ("uske projects batao", PDF),
    ("what is docker", None), ("explain kafka partitions", None),      # fresh topic never inherits a stale file
])
def test_carry_over(q, expected):
    assert cl.carry_over_source(q, STATE, "pdf") == expected


# ---- 4. catalog resolves WHICH FILE the user means (built at ingest, no query-time LLM) ----
CATALOG = [
    {"file": "rashmi_photo.jpg", "modality": "image", "doc_type": "photo", "owner": "Rashmi", "topics": ["portrait"],
     "summary": "portrait photo of a woman", "has_text": False},
    {"file": "Rashmi_Barethiya_19-05.pdf", "modality": "pdf", "doc_type": "resume", "owner": "Rashmi Barethiya",
     "topics": ["technical skills", "projects", "education", "data engineering", "certifications"],
     "summary": "MCA graduate resume with data engineering projects", "has_text": True},
    {"file": "Cracking the Tech Career.pdf", "modality": "pdf", "doc_type": "book", "owner": "",
     "topics": ["resume writing", "cover letters", "interviews", "career advice"],
     "summary": "book of career advice for tech job seekers", "has_text": True},
    {"file": "IT_Direct_Hire_Companies_2026.xlsx", "modality": "spreadsheet", "doc_type": "company list",
     "owner": "", "topics": ["companies", "roles", "ctc", "cities"], "summary": "IT companies hiring freshers in Indian cities", "has_text": True},
]


def files(q, needs_text=True):
    return [e["file"] for e in sc.resolve_by_catalog(q, CATALOG, needs_text)]


def test_turn1_photo_cannot_answer_skills_so_no_clarification_needed():
    assert files("tell me the technical skills of rashmi?") == ["Rashmi_Barethiya_19-05.pdf"]


def test_turn3_the_resume_means_the_resume_not_the_book_about_resumes():
    assert files("tell me the technical skills from the resume.") == ["Rashmi_Barethiya_19-05.pdf"]


def test_book_and_photo_and_sheet_routing():
    assert files("give me interview tips from the book") == ["Cracking the Tech Career.pdf"]
    assert files("show me rashmi photo", needs_text=False) == ["rashmi_photo.jpg"]
    assert files("which company list has ctc info") == ["IT_Direct_Hire_Companies_2026.xlsx"]


def test_unknown_topic_falls_back_to_normal_retrieval():
    assert files("explain kafka consumer groups") == []


def test_catalog_json_parser_tolerates_chatty_models():
    raw = 'Sure! Here you go:\n```json\n{"doc_type": "Resume", "owner": null, "topics": ["skills"], "summary": "x", "has_text": true}\n```'
    assert sc.parse_catalog_json(raw)["doc_type"] == "Resume"
    assert sc.parse_catalog_json("no json here") == {}
