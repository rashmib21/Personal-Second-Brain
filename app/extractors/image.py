import logging
from PIL import Image
import pytesseract
import cv2

from app.rag.image_summarizer import summarize_image

logger = logging.getLogger(__name__)

def extract_image(path):
    """
    Extracts information from an image file.
    Includes:
    - Standard OCR text using Tesseract
    - VLM (Visual Language Model) summary and transcription using Qwen2.5-VL
    - Face detection using OpenCV
    - Image metadata
    """
    try:
        # 1. Open image and run Tesseract OCR
        image = Image.open(path)
        ocr_text = pytesseract.image_to_string(image) or ""
        ocr_text = ocr_text.strip()

        # 2. Generate VLM Summary & Transcription for rich visual content
        vlm_summary = ""
        try:
            vlm_summary = summarize_image(path) or ""
            vlm_summary = vlm_summary.strip()
        except Exception as vlm_err:
            logger.warning(f"VLM summarization skipped or failed for {path}: {vlm_err}")

        # 3. Convert image to OpenCV format for face detection
        image_cv = cv2.imread(path)
        face_count = 0
        faces_data = []

        if image_cv is not None:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
            # OpenCV built-in face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            face_count = len(faces)

            # Store information about each detected face
            for index, (x, y, w, h) in enumerate(faces):
                faces_data.append({
                    "face_id": index,
                    "bbox": [int(x), int(y), int(w), int(h)]
                })

        # 4. Build combined searchable text for vector indexing
        extracted_parts = []

        if vlm_summary and not vlm_summary.startswith("No usable visual information"):
            extracted_parts.append(f"Visual Summary & Transcription:\n{vlm_summary}")

        if ocr_text:
            extracted_parts.append(f"OCR Text:\n{ocr_text}")

        extracted_parts.append(f"Faces Detected: {face_count}")    
        final_text = "\n\n".join(extracted_parts)

        # 5. Return structured result
        return {
            "text": final_text,
            "status": "SUCCESS",
            "metadata": {
                "file_type": "image",
                "ocr_text": ocr_text,
                "vlm_summary": vlm_summary,
                "face_count": face_count,
                "faces": faces_data
            }
        }        

    except Exception as e:
        logger.exception(f"Failed to process image {path}: {e}")
        return {
            "text": "",
            "status": "ERROR",
            "error": str(e)
        }