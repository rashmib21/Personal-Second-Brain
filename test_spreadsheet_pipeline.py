import sys
import os
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.rag.rag_pipeline import ask
from app.services.interaction_state import clear_last_interaction


class TestSpreadsheetProductionPipeline(unittest.TestCase):

    def setUp(self):
        clear_last_interaction()

    def test_01_direct_city_filtering_and_records(self):
        """Verify production ask() returns all records for a valid filter query."""
        res = ask("list the companies of Indore", return_structured=True)
        self.assertIn("answer", res)
        ans = res["answer"]
        lines = [line.strip() for line in ans.split("\n") if line.strip()]
        
        self.assertGreaterEqual(len(lines), 20)
        self.assertTrue(any("Infobeans" in line for line in lines))
        self.assertTrue(any("Systematix" in line for line in lines))
        self.assertEqual(res.get("num_chunks"), 21)

    def test_02_ungrounded_concept_clarification(self):
        """Verify production ask() triggers ungrounded clarification step for unknown fields."""
        res = ask("service based companies", return_structured=True)
        self.assertIn("answer", res)
        ans = res["answer"].lower()
        self.assertTrue(
            "can't determine" in ans or "couldn't find" in ans or "unsupported" in ans,
            f"Expected clarification message, got: {res['answer']}"
        )

    def test_03_numeric_ranking_top_limit(self):
        """Verify production ask() executes user-requested ranking limit."""
        res = ask("top 5 companies by CTC", return_structured=True)
        self.assertIn("answer", res)
        ans = res["answer"]
        lines = [line.strip() for line in ans.split("\n") if line.strip()]
        self.assertEqual(len(lines), 5)

    def test_04_count_query(self):
        """Verify production ask() count query execution."""
        res = ask("how many companies are in Indore?", return_structured=True)
        self.assertIn("answer", res)
        ans = res["answer"]
        self.assertTrue("21" in ans or "19" in ans or "20" in ans, f"Got: {ans}")

    def test_05_explicit_source_spec(self):
        """Verify production ask() handles explicit source file query."""
        res = ask("From IT_Direct_Hire_Companies_2026.xlsx, list the companies in Indore", return_structured=True)
        self.assertIn("answer", res)
        ans = res["answer"]
        lines = [line.strip() for line in ans.split("\n") if line.strip()]
        self.assertGreaterEqual(len(lines), 20)

    def test_06_non_spreadsheet_rag(self):
        """Verify non-spreadsheet RAG queries pass through unchanged."""
        res = ask("Where did I mention Kafka?", return_structured=True)
        self.assertIn("answer", res)
        self.assertIn("sources", res)
        self.assertGreater(len(res["answer"]), 10)


if __name__ == "__main__":
    unittest.main()
