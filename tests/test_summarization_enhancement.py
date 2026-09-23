"""
Unit and integration tests for document summarization enhancement and TOC noise filtering.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.search.summary_search import is_toc_chunk, get_clean_document_summary_context


class TestSummarizationEnhancement(unittest.TestCase):

    def test_is_toc_chunk_detection(self):
        # 1. Table of contents chunks should return True
        self.assertTrue(is_toc_chunk("Table of Contents\nChapter 1 Getting Started ... 5\nChapter 2 Advanced ... 12"))
        self.assertTrue(is_toc_chunk("Chapters 1-62: Various authors (e.g. Bart Schuijt, Jaydip Jadhav)"))
        self.assertTrue(is_toc_chunk("Contributors: John Doe, Jane Smith, Alex Johnson"))
        self.assertTrue(is_toc_chunk("GoalKicker.com - Free Programming Books"))

        # 2. Real technical body chunks should return False
        body_sample_1 = (
            "SQL SELECT statement is used to fetch data from a database table. "
            "Syntax: SELECT column1, column2 FROM table_name WHERE condition; "
            "Indexes speed up data retrieval operations on database tables at the cost of additional writes."
        )
        self.assertFalse(is_toc_chunk(body_sample_1))

        body_sample_2 = (
            "Transactions in relational databases ensure ACID properties: Atomicity, Consistency, Isolation, and Durability. "
            "Use COMMIT to make changes permanent and ROLLBACK to undo changes."
        )
        self.assertFalse(is_toc_chunk(body_sample_2))

    @patch("app.search.summary_search.get_table")
    def test_get_clean_document_summary_context_sampling(self, mock_get_table):
        # Create a mock LanceDB table with TOC chunks and body chunks
        mock_table = MagicMock()
        
        data = []
        # Chunks 0..4: TOC / metadata
        data.append({"chunk_id": "doc_chunk_0", "path": "/watched_folder/SQLNotesForProfessionals.pdf", "file_type": "pdf", "text": "Table of Contents\nChapter 1 ... 1\nChapter 2 ... 50"})
        data.append({"chunk_id": "doc_chunk_1", "path": "/watched_folder/SQLNotesForProfessionals.pdf", "file_type": "pdf", "text": "Chapters 1-62: Various authors (Bart Schuijt, Jaydip Jadhav)"})
        
        # Chunks 2..50: Body chunks about SQL topics
        topics = ["SELECT Queries", "JOINs and Union", "Indexes and B-Trees", "Transactions and Locks", "Window Functions", "Stored Procedures", "Triggers", "Views and CTEs"]
        for i in range(2, 52):
            topic = topics[i % len(topics)]
            text_content = f"Chapter {i}: Detailed technical concepts on {topic}. " * 50
            data.append({"chunk_id": f"doc_chunk_{i}", "path": "/watched_folder/SQLNotesForProfessionals.pdf", "file_type": "pdf", "text": text_content})
            
        mock_df = pd.DataFrame(data)
        mock_table.to_pandas.return_value = mock_df
        mock_get_table.return_value = mock_table

        context_str, total_chunks, sampled_count = get_clean_document_summary_context("/watched_folder/SQLNotesForProfessionals.pdf", max_chars=10000)

        # Assertions:
        # 1. Total chunks recognized is 52
        self.assertEqual(total_chunks, 52)
        # 2. Context string does NOT contain TOC / author noise
        self.assertNotIn("Table of Contents", context_str)
        self.assertNotIn("Various authors", context_str)
        # 3. Context string contains sampled body concepts and is non-empty
        self.assertGreater(sampled_count, 0)
        self.assertIn("Indexes and B-Trees", context_str)


if __name__ == "__main__":
    unittest.main()
