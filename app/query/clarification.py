"""
Generic conversation helpers (documents, images, audio, spreadsheets, ...).
No domain words, no file names. Pure functions, easy to test.

  resolve_selection(reply, candidates)  -> the candidate the user picked, or None
  carry_over_source(question, state)    -> last active source for follow-ups like 'what are his projects there'
"""
import os
import re
import difflib

_FILLER = {
    "the", "a", "an", "one", "ones", "file", "files", "please", "pls", "ok", "okay", "yes", "yeah", "i", "mean",
    "meant", "want", "wanted", "this", "that", "it", "is", "was", "use", "using", "from", "in", "of", "wala",
    "wali", "wale", "waali", "hi", "sorry", "actually", "select", "choose", "pick", "open", "go", "with",
}
_ORDINALS = {
    "first": 0, "1st": 0, "1": 0, "pehla": 0, "pehle": 0, "pahla": 0, "pahle": 0,
    "second": 1, "2nd": 1, "2": 1, "dusra": 1, "doosra": 1, "dusre": 1,
    "third": 2, "3rd": 2, "3": 2, "teesra": 2, "tisra": 2,
    "last": -1, "latter": -1, "aakhri": -1,
}
_TYPE_WORDS = {
    "pdf": {".pdf"},
    "docx": {".docx", ".doc"}, "word": {".docx", ".doc"},
    "txt": {".txt", ".md"}, "text": {".txt", ".md"},
    "document": {".pdf", ".docx", ".doc", ".txt", ".md"}, "doc": {".pdf", ".docx", ".doc", ".txt", ".md"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "photo": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "picture": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "pic": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "jpg": {".jpg", ".jpeg"}, "jpeg": {".jpg", ".jpeg"}, "png": {".png"}, "webp": {".webp"},
    "excel": {".xlsx", ".xls", ".ods"}, "xlsx": {".xlsx"}, "xls": {".xls"}, "csv": {".csv"},
    "spreadsheet": {".xlsx", ".xls", ".csv", ".ods"}, "sheet": {".xlsx", ".xls", ".csv", ".ods"},
    "audio": {".mp3", ".m4a", ".wav", ".mpeg"}, "recording": {".mp3", ".m4a", ".wav", ".mpeg"},
    "video": {".mp4", ".mkv", ".mov", ".avi"},
}


def _stem_tokens(name):
    base = os.path.splitext(os.path.basename(str(name)))[0].lower()
    return [t for t in re.split(r"[^a-z0-9]+", base) if t]


def _reply_tokens(reply):
    text = reply.strip().strip("\"'`").lower()
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    return text, [t for t in re.split(r"[^a-z0-9\-_.]+", text) if t]


def resolve_selection(reply, candidates):
    """
    The user answered 'which one did you mean?'. Return the chosen candidate (same string as in `candidates`)
    or None when the reply is not an unambiguous pick (e.g. it is a brand-new question).
    """
    if not reply or not candidates:
        return None
    text, toks = _reply_tokens(reply)
    if not toks:
        return None

    # 1. exact file name / stem (extension optional, trailing '.' ignored)
    exact = [c for c in candidates
             if text == os.path.basename(c).lower() or text == os.path.splitext(os.path.basename(c))[0].lower()]
    if len(exact) == 1:
        return exact[0]

    core = [t for t in toks if t not in _FILLER]
    if not core:
        return None

    # 2. ordinal: 'first', '2', 'pehla wala'
    if len(core) == 1 and core[0] in _ORDINALS:
        idx = _ORDINALS[core[0]]
        try:
            return candidates[idx]
        except IndexError:
            return None

    # 3. type words: 'the pdf', 'photo wala', 'excel one'
    type_words = [t for t in core if t in _TYPE_WORDS]
    if type_words and len(type_words) == len(core):
        exts = set.intersection(*[_TYPE_WORDS[t] for t in type_words])
        hit = [c for c in candidates if os.path.splitext(c)[1].lower() in exts]
        if len(hit) == 1:
            return hit[0]
        return None

    # 4. every non-filler word of the reply must be explained by ONE candidate's file-name tokens
    #    (so a real question like 'tell me the technical skills from the resume' never matches)
    hits = []
    for c in candidates:
        ctoks = _stem_tokens(c)
        ext = os.path.splitext(c)[1].lower().lstrip(".")

        def explained(t):
            t = os.path.splitext(t)[0] if t.endswith(("." + ext)) else t
            t = re.sub(r"[.\-_]+$", "", t)
            parts = [p for p in re.split(r"[^a-z0-9]+", t) if p]
            return all(any(p == ct or (len(p) >= 4 and difflib.SequenceMatcher(None, p, ct).ratio() >= 0.85)
                           for ct in ctoks) or p == ext for p in parts)

        if all(explained(t) or t in _TYPE_WORDS and ext in {e.lstrip('.') for e in _TYPE_WORDS[t]} for t in core):
            hits.append(c)
    if len(hits) == 1:
        return hits[0]

    # 5. typo-tolerant whole-name match
    scored = sorted(((difflib.SequenceMatcher(None, text, os.path.basename(c).lower()).ratio(), c) for c in candidates),
                    reverse=True)
    if scored and scored[0][0] >= 0.88 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08):
        return scored[0][1]
    return None


DEICTIC = {
    "it", "its", "this", "that", "there", "his", "her", "their", "same", "above", "earlier", "previous", "prior",
    "uska", "uski", "uske", "isme", "usme", "isse", "usse", "wahi", "yahi", "iska", "iski", "iske",
}


def carry_over_source(question, state, modality_hint=None):
    """
    Follow-up without a named source ('what are his projects there', 'and the education?').
    Returns the last active source of the right modality, else None.
    Only fires when the question has a deictic/possessive word, so a fresh topic never inherits a stale file.
    """
    words = set(re.findall(r"[a-z]+", (question or "").lower()))
    if not (words & DEICTIC):
        return None
    order = ["last_document_source", "last_image_source", "last_spreadsheet_source"]
    if modality_hint in ("image",):
        order = ["last_image_source"]
    elif modality_hint in ("spreadsheet",):
        order = ["last_spreadsheet_source"]
    elif modality_hint in ("pdf", "docx", "document", "text"):
        order = ["last_document_source"]
    for key in order:
        if state.get(key):
            return state[key]
    return state.get("previous_source") or None
