import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.extractors.image import extract_image

image_path = "tests/test_images/test.png"

result = extract_image(image_path)

print("\n========== RESULT ==========")
print("Status:", result["status"])
print("Text:", result["text"])
print("Metadata:", result.get("metadata"))
print("Error:", result.get("error"))
print("============================")