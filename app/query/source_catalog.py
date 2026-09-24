"""
Source catalog: what each indexed file IS (doc_type, owner, topics), built ONCE per file at ingest time
(one LLM call per file, not per query). Queries are then mapped to files by matching this catalog.
No domain words are hardcoded: 'resume' works because the catalog says doc_type='resume', not because of a list.
"""
import json
import os
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_ST = PorterStemmer()
CATALOG_PATH = os.environ.get("SOURCE_CATALOG_PATH", "database/source_catalog.json")

CATALOG_PROMPT = """You describe ONE file for a search index. Reply with JSON only, no prose, no markdown.
Keys:
  "doc_type": short noun for what the file is (e.g. resume, invoice, book, research paper, meeting recording,
              photo, screenshot, company list, sales table). Lowercase, 1-3 words.
  "owner": person/organisation the file is about or belongs to, or null.
  "topics": up to 8 short topics it covers.
  "summary": one sentence.
  "has_text": true if it contains readable text/data that can answer questions, false for pure photos.
File name: {name}
Content sample:
{sample}
"""

_REQUEST = {"tell", "give", "show", "provide", "list", "please", "kindly", "want", "need", "name", "names",
            "batao", "dikhao", "mujhe", "wala", "wali", "wale", "kya", "hai", "hain", "ka", "ki", "ke", "se", "mein"}


def _toks(text):
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(w) > 2 and w not in ENGLISH_STOP_WORDS and w not in _REQUEST]


def _stems(text):
    return {_ST.stem(w) for w in _toks(text)}


def parse_catalog_json(raw):
    """Tolerant JSON extraction (small models wrap JSON in prose/markdown)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def build_entry(path, sample_text, llm_call, modality):
    """llm_call(prompt)->str : Gemini / Groq / Ollama, whatever you use. Called once per file."""
    name = os.path.basename(path)
    d = parse_catalog_json(llm_call(CATALOG_PROMPT.format(name=name, sample=(sample_text or "")[:2500])))
    return {
        "file": name, "path": path, "modality": modality,
        "doc_type": str(d.get("doc_type") or "").lower(),
        "owner": d.get("owner") or "",
        "topics": [str(t) for t in (d.get("topics") or [])][:8],
        "summary": str(d.get("summary") or ""),
        "has_text": bool(d.get("has_text", modality != "image")),
    }


def load_catalog(path=CATALOG_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_entry(entry, path=CATALOG_PATH):
    cat = [e for e in load_catalog(path) if e.get("file") != entry["file"]]
    cat.append(entry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=1)


WEIGHTS = {"doc_type": 5.0, "owner": 4.0, "file": 3.0, "topics": 1.0, "summary": 0.5}


def score_entry(question_stems, entry):
    fields = {
        "doc_type": _stems(entry.get("doc_type", "")),
        "owner": _stems(entry.get("owner", "")),
        "file": _stems(os.path.splitext(entry.get("file", ""))[0].replace("_", " ")),
        "topics": _stems(" ".join(entry.get("topics", []))),
        "summary": _stems(entry.get("summary", "")),
    }
    return sum(WEIGHTS[k] * len(question_stems & v) for k, v in fields.items())


def resolve_by_catalog(question, entries, needs_text=True, min_score=3.0):
    """
    Returns [] (no confident source -> use normal retrieval), [one] (resolved), or [several] (ask the user).
    needs_text=True drops pure photos: a portrait cannot answer 'technical skills'.
    """
    qs = _stems(question)
    pool = [e for e in entries if e.get("has_text", True)] if needs_text else list(entries)
    scored = sorted(((score_entry(qs, e), e) for e in pool), key=lambda x: -x[0])
    scored = [(s, e) for s, e in scored if s >= min_score]
    if not scored:
        return []
    top = scored[0][0]
    close = [e for s, e in scored if s >= 0.6 * top]
    return close
