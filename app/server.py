import os
import sys
import shutil
import logging
from pathlib import Path

# Suppress HTTP request logs from external clients (httpx, httpcore, urllib3)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Ensure project root is in sys.path when running app/server.py directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import existing backend modules
from app.rag.rag_pipeline import ask
from app.storage.lancedb_store import (
    get_table,
    get_image_table,
    get_face_table,
    get_hash_table,
    total_chunks,
    total_files,
    total_faces
)
from app.services.face_service import register_face, find_faces_for_person
from app.extractors.router import extract_file
from app.utils.file_inventory import inventory, get_watched_folder

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHED_FOLDER = get_watched_folder()
WATCHED_FOLDER.mkdir(parents=True, exist_ok=True)



app = Flask(__name__)

# Enable CORS for all origins and routes
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"}), 200


# ==========================================
# 1. SYSTEM STATS & STATUS
# ==========================================

@app.route("/api/stats", methods=["GET"])
def get_system_stats():
    """Returns database metrics and backend service statuses."""
    try:
        t_chunks = total_chunks()
        t_files = total_files()
        t_faces = total_faces()

        # Count images in image table if available
        try:
            img_table = get_image_table()
            t_images = img_table.count_rows()
        except Exception:
            t_images = 0

        # Model status check
        status = {
            "rag": "Ready",
            "embeddings": "Active (BGE-Small / CLIP 512D)",
            "image_search": "Active (Qwen2.5-VL)",
            "face_search": "Active (FaceNet 512D)",


            "files_indexed": t_files,
            "document_chunks": t_chunks,
            "images_indexed": t_images,
            "faces_indexed": t_faces,
        }
        return jsonify(status), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


# ==========================================
# 2. CHAT & RAG PIPELINE
# ==========================================

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    Executes the multimodal RAG pipeline for a user question.
    Expected payload: {"question": "...", "active_file": "...", "modality": "..."}
    """
    payload = request.get_json(silent=True) or {}
    question_text = payload.get("question", "").strip()
    active_file = (
        payload.get("active_file")
        or payload.get("source")
        or payload.get("file")
        or payload.get("activeFile")
    )
    modality = (
        payload.get("modality")
        or payload.get("scope")
    )
    if not question_text:
        return jsonify({"detail": "Question cannot be empty."}), 400

    try:
        response_data = ask(
            question_text,
            active_file=active_file,
            modality=modality,
            return_structured=True
        )
        return jsonify(response_data), 200
    except Exception as error_exception:
        return jsonify({"detail": f"RAG Error: {str(error_exception)}"}), 500


# ==========================================
# 3. FILE MANAGEMENT & MEDIA SERVING
# ==========================================

@app.route("/api/files", methods=["GET"])
def list_files():
    """Return the authoritative current filesystem inventory."""
    try:
        files = []
        for row in inventory():
            files.append({
                "filename": row["filename"],
                "path": row["path"],
                "file_type": row["category"],
                "extension": row["extension"],
                "size": row["size"],
                "modified_at": row["mtime"],
            })
        return jsonify({"files": files}), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


@app.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": [r for r in list_files_inventory("document", "text")]})


@app.route("/api/spreadsheets", methods=["GET"])
def list_spreadsheets():
    return jsonify({"spreadsheets": list_files_inventory("spreadsheet")})


@app.route("/api/images", methods=["GET"])
def list_images():
    return jsonify({"images": list_files_inventory("image")})


def list_files_inventory(*categories):
    allowed = set(categories)
    return [
        {
            "filename": row["filename"],
            "path": row["path"],
            "file_type": row["category"],
            "extension": row["extension"],
            "size": row["size"],
            "modified_at": row["mtime"],
        }
        for row in inventory()
        if row["category"] in allowed
    ]


@app.route("/api/media/<path:file_path>", methods=["GET"])
def serve_media(file_path: str):
    """Serve only files that are inside the watched folder."""
    try:
        requested = (WATCHED_FOLDER / file_path).resolve()
        root = WATCHED_FOLDER.resolve()
        if root not in requested.parents and requested != root:
            return jsonify({"detail": "Invalid media path."}), 403
        if not requested.is_file():
            return jsonify({"detail": "File not found."}), 404
        return send_file(str(requested))
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload safely into watched_folder and publish through the normal pipeline."""
    try:
        uploaded_file = request.files.get("file")
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({"detail": "No file uploaded."}), 400

        filename = secure_filename(uploaded_file.filename)
        if not filename:
            return jsonify({"detail": "Invalid filename."}), 400

        dest_path = (WATCHED_FOLDER / filename).resolve()
        if WATCHED_FOLDER.resolve() not in dest_path.parents:
            return jsonify({"detail": "Invalid upload path."}), 400

        uploaded_file.save(str(dest_path))
        result = extract_file(str(dest_path))
        return jsonify({
            "status": "SUCCESS",
            "filename": filename,
            "path": str(dest_path),
            "result": result
        }), 200
    except Exception as error_exception:
        return jsonify({"detail": f"Upload processing failed: {error_exception}"}), 500


# ==========================================
# 4. FACE MEMORY APIs
# ==========================================

@app.route("/api/faces", methods=["GET"])
def get_faces():
    """Lists registered face memory identities and all detected face records."""
    try:
        ftable = get_face_table()
        df = ftable.to_pandas()
        if df.empty:
            return jsonify({"faces": [], "persons": []}), 200

        records = df.to_dict(orient="records")
        persons = list(set(df["person_name"].dropna().tolist()))

        formatted_faces = []
        for r in records:
            path_str = r.get("image_path", "")
            raw_bbox = r.get("bbox")
            if raw_bbox is not None and hasattr(raw_bbox, "tolist"):
                bbox = raw_bbox.tolist()
            elif raw_bbox is not None:
                bbox = list(raw_bbox)
            else:
                bbox = [0, 0, 0, 0]

            formatted_faces.append({
                "face_id": str(r.get("face_id")),
                "person_name": str(r.get("person_name", "unknown")),
                "image_path": str(path_str),
                "filename": os.path.basename(str(path_str)),
                "bbox": bbox,
                "created_at": str(r.get("created_at"))
            })

        return jsonify({
            "faces": formatted_faces,
            "persons": persons
        }), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


@app.route("/api/faces/register", methods=["POST"])
def register_face_endpoint():
    """
    Registers a reference face for a person given an image path or uploaded file.
    """
    try:
        person_name = request.form.get("person_name")
        image_path = request.form.get("image_path")
        file = request.files.get("file")

        target_path = image_path
        if file:
            dest_path = WATCHED_FOLDER / file.filename
            file.save(str(dest_path))
            target_path = str(dest_path)

        if not target_path or not person_name:
            return jsonify({"detail": "Either image_path or file, and person_name must be provided."}), 400

        result = register_face(person_name=person_name, image_path=target_path)
        return jsonify(result), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
