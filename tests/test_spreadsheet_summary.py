import unittest
import os
import pandas as pd
from app.rag.spreadsheet_query import (
    generate_spreadsheet_summary,
    inspect_sheet_schema
)
from config import SPREADSHEET_MODEL_NAME


class TestSpreadsheetSummary(unittest.TestCase):

    def test_spreadsheet_model_config(self):
        """Test that SPREADSHEET_MODEL_NAME is configured."""
        self.assertIsNotNone(SPREADSHEET_MODEL_NAME)
        self.assertTrue(len(SPREADSHEET_MODEL_NAME) > 0)

    def test_generate_spreadsheet_summary_output(self):
        """Test statistical profiling and summary generation for structured sheets."""
        sample_df = pd.DataFrame({
            "Company": ["TechCorp", "DataInc", "CloudSoft"],
            "City": ["Indore", "Indore", "Bhopal"],
            "Employees": [100, 200, 150]
        })

        sheets = {
            "Sheet1": {
                "df": sample_df,
                "schema": inspect_sheet_schema(sample_df, "Sheet1")
            }
        }

        answer, total_rows = generate_spreadsheet_summary("dummy_companies.xlsx", sheets, "Summarize dummy_companies.xlsx")

        self.assertEqual(total_rows, 3)
        self.assertIn("Spreadsheet Overview", answer)
        self.assertIn("Sheet: 'Sheet1' (3 total records)", answer)
        self.assertIn("Company, City, Employees", answer)
        self.assertIn("Employees", answer)


if __name__ == "__main__":
    unittest.main()
