import unittest
from app.chunker.chunker import chunk_text, split_text_recursively


class TestRecursiveChunker(unittest.TestCase):

    def test_empty_or_whitespace_input(self):
        """Test that empty or whitespace-only inputs return empty list."""
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])
        self.assertEqual(chunk_text(None), [])

    def test_short_text_single_chunk(self):
        """Test that text smaller than max_chars returns a single chunk without duplication."""
        text = "Tata Communications provides global connectivity and cloud security solutions."
        chunks = chunk_text(text, max_chars=600)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_multi_paragraph_sliding_overlap(self):
        """Test multi-paragraph text splitting with sliding overlap."""
        paragraph1 = "Paragraph 1: " + ("Word " * 40)
        paragraph2 = "Paragraph 2: " + ("Data " * 40)
        paragraph3 = "Paragraph 3: " + ("Info " * 40)
        full_text = f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

        chunks = chunk_text(full_text, max_chars=300, overlap_chars=80)

        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)

        # Check that no chunk exceeds max_chars
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 300)

        # Check that adjacent chunks have overlap (context continuity)
        first_chunk_end = chunks[0][-50:]
        self.assertTrue(first_chunk_end in chunks[1] or chunks[1].startswith(chunks[0].split("\n")[-1]))

    def test_long_single_paragraph_sentence_splitting(self):
        """Test long unbroken paragraph splits recursively by sentence boundaries."""
        sentences = [f"Sentence number {i} is detailing key requirements." for i in range(1, 20)]
        long_paragraph = " ".join(sentences)

        chunks = chunk_text(long_paragraph, max_chars=250, overlap_chars=60)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 250)

    def test_image_ocr_text_chunking(self):
        """Test chunking of image OCR / VLM summary text."""
        ocr_text = "Visual Summary & Transcription:\nReceipt from Acme Store on Jan 2026.\n\nOCR Text:\nItem 1: Server Rack $1200\nItem 2: Router $300\nTotal: $1500"
        chunks = chunk_text(ocr_text, max_chars=600)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Visual Summary", chunks[0])


if __name__ == "__main__":
    unittest.main()
