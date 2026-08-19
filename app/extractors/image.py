import logging
from PIL import Image
import pytesseract
import cv2

logger = logging.getLogger(__name__)

def extract_image(path):
    """
    Extracts information from image.
    includes: ocr text using Tesseract, face detection using opencv, basic image metadata 
    """
    try:
        #open image
        image = Image.open(path)
        #convert
        text = pytesseract.image_to_string(image) or ""
        text = text.strip()

        #Convert image to OpenCV format
        image_cv=cv2.imread(path)
        face_count=0

        if image_cv is not None:
            gray=cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
            #Opencv built-in face detector
            face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            faces=face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30)
                )
            face_count=len(faces)

        #Build searchable text
        extracted_parts=[]

        if text:
            extracted_parts.append(f"OCR Text:\n{text}")
        extracted_parts.append(f"Faces Detected: {face_count}")    
        final_text="\n\n".join(extracted_parts)

        #Retun structured result
        return {
            "text":final_text,
            "status":"SUCCESS",
            "metadata":{
                    "file_type":"image",
                    "ocr_text":text,
                    "face_count":face_count
            }
        }        

    
    except Exception as e:
        logger.exception(f"Failed to process image {path}: {e}")
        return {
            "text": "",
            "status": "ERROR",
            "error": str(e)
        }