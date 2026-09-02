import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import logging
import os

logger = logging.getLogger(__name__)

# Singleton lazy-loaded model instance
_FACE_MODEL = None

def get_face_model():
    """
    Returns the singleton InceptionResnetV1 pretrained face embedding model.
    """
    global _FACE_MODEL
    if _FACE_MODEL is None:
        logger.info("Loading InceptionResnetV1 (vggface2) face embedding model...")
        _FACE_MODEL = InceptionResnetV1(pretrained='vggface2').eval()
    return _FACE_MODEL


# Standard PyTorch transform for FaceNet (160x160 normalized to [-1, 1])
_FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def detect_and_embed_faces(image_path):
    """
    Detects faces in an image using OpenCV Haar Cascade, crops each face,
    and extracts a normalized 512-D face feature embedding.

    Returns:
        list of dicts: [
            {
                "face_id": int,
                "bbox": [x, y, w, h],
                "embedding": list of 512 floats
            }, ...
        ]
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image path does not exist: {image_path}")
        return []

    image_cv = cv2.imread(image_path)
    if image_cv is None:
        logger.warning(f"Could not load image with cv2: {image_path}")
        return []

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return []

    model = get_face_model()
    face_results = []

    for index, (x, y, w, h) in enumerate(faces):
        # Add slight margin padding around bounding box if possible
        img_h, img_w = image_cv.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        face_crop_bgr = image_cv[y1:y2, x1:x2]
        if face_crop_bgr.size == 0:
            continue

        # Convert BGR crop to RGB PIL Image
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(face_crop_rgb)

        # Preprocess tensor and generate embedding
        tensor_crop = _FACE_TRANSFORM(pil_crop).unsqueeze(0)

        with torch.no_grad():
            embedding_tensor = model(tensor_crop)
            # L2 Normalize
            embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
            embedding_list = embedding_tensor[0].cpu().numpy().tolist()

        face_results.append({
            "face_id": index,
            "bbox": [int(x), int(y), int(w), int(h)],
            "embedding": embedding_list
        })

    return face_results
