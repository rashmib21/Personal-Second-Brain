"""
Regression test suite for the IntentRouter integration.
Tests A-R from the approved implementation plan.

Run with:
    python -m pytest tests/test_intent_routing.py -v
or:
    python tests/test_intent_routing.py
"""

import sys
import os
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.query.query_plan import QueryIntent, RequestScope, Modality
from app.query.query_analyzer import analyze_query

AUDIO_FILES = ["Behari_lal_call.m4a", "conference_recording.mp3"]
IMAGE_FILES = ["mummy.jpg", "dense.webp", "dense2.webp", "photo_001.png"]
DOC_FILES   = ["SQLNotesForProfessionals.pdf", "PhysicsNotes.pdf", "report.docx"]
ALL_FILES   = AUDIO_FILES + IMAGE_FILES + DOC_FILES


def get_plan(query, indexed_files=None):
    files = indexed_files if indexed_files is not None else ALL_FILES
    with patch("app.query.query_analyzer.get_indexed_filenames", return_value=files), \
         patch("app.query.query_analyzer.resolve_semantic_source", return_value=(None, 0.0, 0.0, [])), \
         patch("app.query.query_analyzer.resolve_canonical_source_id", side_effect=lambda h: h), \
         patch("app.storage.lancedb_store.get_hash_table", return_value=None):
        result = analyze_query(query, indexed_files=files)
        return result["plan"]


class TestIntentRouting(unittest.TestCase):

    # A. Same intent, different wording
    def test_A1_image_display_show_me(self):
        plan = get_plan("show me the photo of mummy", IMAGE_FILES)
        self.assertEqual(plan.intent, QueryIntent.IMAGE_DISPLAY, f"Got: {plan.intent}")

    def test_A2_image_display_display(self):
        plan = get_plan("display the picture", IMAGE_FILES)
        self.assertEqual(plan.intent, QueryIntent.IMAGE_DISPLAY, f"Got: {plan.intent}")

    def test_A3_image_display_i_want_to_see(self):
        plan = get_plan("I want to see the image", IMAGE_FILES)
        self.assertEqual(plan.intent, QueryIntent.IMAGE_DISPLAY, f"Got: {plan.intent}")

    def test_A4_image_display_can_you_show(self):
        plan = get_plan("can you show me the photo", IMAGE_FILES)
        self.assertEqual(plan.intent, QueryIntent.IMAGE_DISPLAY, f"Got: {plan.intent}")

    # B. Natural-language source reference
    def test_B_sql_notes_source_resolved(self):
        plan = get_plan("summarize the SQL notes", DOC_FILES)
        self.assertIsNotNone(plan.source_spec.source_hint, "source_hint must not be None")
        hint = (plan.source_spec.source_hint or "").upper()
        self.assertIn("SQL", hint, f"Expected SQL in source hint, got: {hint}")

    # C. Unknown source -> explicit + unresolved
    def test_C_unknown_source_physics_notes(self):
        files = ["SQLNotesForProfessionals.pdf", "report.docx"]
        plan = get_plan("summarize the physics notes", files)
        self.assertTrue(plan.source_spec.is_explicit,
                        "Must be is_explicit=True for explicitly named but unindexed file")
        self.assertFalse(plan.source_spec.is_resolved,
                         "Must be is_resolved=False when file is absent from index")

    # E. Cross-modality negative: QA must NOT become SUMMARIZATION
    def test_E_normal_qa_stays_qa(self):
        plan = get_plan("What did the SQL notes say about indexing?", DOC_FILES)
        self.assertEqual(plan.intent, QueryIntent.QUESTION_ANSWERING, f"Got: {plan.intent}")

    # F. Metadata inventory
    def test_F1_list_all_files(self):
        plan = get_plan("list all files")
        self.assertEqual(plan.intent, QueryIntent.METADATA_QUERY, f"Got: {plan.intent}")

    def test_F2_how_many_files(self):
        plan = get_plan("how many files do I have?")
        self.assertEqual(plan.intent, QueryIntent.METADATA_QUERY, f"Got: {plan.intent}")

    def test_F3_show_all_audio(self):
        plan = get_plan("show all audio recordings")
        self.assertEqual(plan.intent, QueryIntent.METADATA_QUERY, f"Got: {plan.intent}")
        self.assertEqual(plan.modality, Modality.AUDIO, f"Modality: {plan.modality}")

    # G. Date filtering
    def test_G_date_filter_populated(self):
        plan = get_plan("files added on 12th September")
        self.assertEqual(plan.intent, QueryIntent.METADATA_QUERY, f"Got: {plan.intent}")
        self.assertIsNotNone(plan.filters.start_datetime,
                             "start_datetime should be set for date queries")

    # H. Latest/recent no N -> default 5
    def test_H_recent_no_n(self):
        plan = get_plan("show me recent files")
        self.assertEqual(plan.intent, QueryIntent.METADATA_QUERY, f"Got: {plan.intent}")
        self.assertEqual(plan.filters.extracted_limit, 5,
                         "Default limit for 'recent' must be 5")

    # I. Latest/recent with explicit N
    def test_I_recent_explicit_n(self):
        plan = get_plan("show me the last 3 audio files")
        self.assertEqual(plan.intent, QueryIntent.METADATA_QUERY, f"Got: {plan.intent}")
        self.assertEqual(plan.filters.extracted_limit, 3, "extracted_limit must be 3")
        self.assertEqual(plan.modality, Modality.AUDIO, f"Modality: {plan.modality}")

    # J. Complete extraction
    def test_J_complete_extraction(self):
        plan = get_plan("give me the complete transcript of Behari_lal_call.m4a", AUDIO_FILES)
        self.assertEqual(plan.intent, QueryIntent.FULL_CONTENT_FETCH, f"Got: {plan.intent}")
        self.assertEqual(plan.scope, RequestScope.COMPLETE_FILE, f"Scope: {plan.scope}")

    # K. Complete translation
    def test_K_complete_translation(self):
        plan = get_plan("translate the entire audio to English", AUDIO_FILES)
        self.assertEqual(plan.intent, QueryIntent.FULL_CONTENT_FETCH, f"Got: {plan.intent}")
        self.assertEqual(plan.filters.target_language, "english",
                         "target_language must be 'english'")

    # L. Document summary
    def test_L_document_summary(self):
        plan = get_plan("summarize the report", DOC_FILES)
        self.assertEqual(plan.intent, QueryIntent.SUMMARIZATION, f"Got: {plan.intent}")

    # M. Image display -> IMAGE_DISPLAY
    def test_M_image_display_intent(self):
        plan = get_plan("show me the photo of mummy", IMAGE_FILES)
        self.assertEqual(plan.intent, QueryIntent.IMAGE_DISPLAY, f"Got: {plan.intent}")
        self.assertEqual(plan.modality, Modality.IMAGE, f"Modality: {plan.modality}")

    # N. Image summary -> VISUAL_QA or SUMMARIZATION
    def test_N_image_summary_describe(self):
        plan = get_plan("describe what is in the picture", IMAGE_FILES)
        self.assertIn(plan.intent, [QueryIntent.VISUAL_QA, QueryIntent.SUMMARIZATION],
                      f"Got: {plan.intent}")

    def test_N_image_summary_summarize(self):
        plan = get_plan("summarize the image", IMAGE_FILES)
        self.assertIn(plan.intent, [QueryIntent.VISUAL_QA, QueryIntent.SUMMARIZATION],
                      f"Got: {plan.intent}")

    # O. Audio summary -> SUMMARIZATION + AUDIO
    def test_O_audio_summary_discussed(self):
        plan = get_plan("what was discussed in the call?", AUDIO_FILES)
        self.assertEqual(plan.intent, QueryIntent.SUMMARIZATION, f"Got: {plan.intent}")
        self.assertEqual(plan.modality, Modality.AUDIO, f"Modality: {plan.modality}")

    # P. Normal QA -> QUESTION_ANSWERING
    def test_P_normal_qa(self):
        plan = get_plan("What is the indexing strategy described in the SQL notes?", DOC_FILES)
        self.assertEqual(plan.intent, QueryIntent.QUESTION_ANSWERING, f"Got: {plan.intent}")

    def test_P_open_factual_question(self):
        plan = get_plan("What is gradient descent?", DOC_FILES)
        self.assertEqual(plan.intent, QueryIntent.QUESTION_ANSWERING, f"Got: {plan.intent}")

    # Q. Chapter/structure
    def test_Q_chapter_list(self):
        plan = get_plan("list all chapters in the book", DOC_FILES)
        self.assertIn(plan.intent,
                      [QueryIntent.FULL_CONTENT_FETCH, QueryIntent.METADATA_QUERY],
                      f"Got: {plan.intent}")

    # R. Multi-intent: source + scope + language
    def test_R_multi_intent_translate(self):
        plan = get_plan("translate the complete Behari call to English", AUDIO_FILES)
        self.assertEqual(plan.intent, QueryIntent.FULL_CONTENT_FETCH, f"Got: {plan.intent}")
        self.assertEqual(plan.scope, RequestScope.COMPLETE_FILE, f"Scope: {plan.scope}")
        self.assertEqual(plan.filters.target_language, "english",
                         "target_language must be 'english'")
        self.assertEqual(plan.modality, Modality.AUDIO, f"Modality: {plan.modality}")

    # Anaphora: "the call" must NOT trigger pronouns
    def test_the_call_not_anaphora(self):
        plan = get_plan("summarize the call", AUDIO_FILES)
        self.assertEqual(plan.modality, Modality.AUDIO, "Modality must be AUDIO")
        # the source should not be resolved purely from a history lookup for "the call"
        # (it either resolves via semantic search or stays None — both are correct)

    # Correction: no hardcoded source
    def test_correction_no_hardcoded_source(self):
        plan = get_plan("you are wrong", ALL_FILES)
        self.assertEqual(plan.intent, QueryIntent.CORRECTION, f"Got: {plan.intent}")
        if plan.source_spec.source_hint:
            self.assertNotEqual(plan.source_spec.source_hint.lower(), "mummy.jpg",
                                "Hardcoded 'mummy.jpg' must not appear as source_hint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
