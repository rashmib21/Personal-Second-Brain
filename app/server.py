import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List

# Suppress HTTP request logs from external clients (httpx, httpcore, urllib3)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

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

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHED_FOLDER = BASE_DIR / "watched_folder"
WATCHED_FOLDER.mkdir(exist_ok=True)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static"
)

# Enable CORS for all origins and routes
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def read_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    return jsonify({"message": "Personal Second Brain API is running."}), 200


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
            "audio_transcription": "Active (Srota ASR bfloat16)",
            "video_processing": "Active",
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
    Expected payload: {"question": "..."}
    """
    payload = request.get_json(silent=True) or {}
    question_text = payload.get("question", "").strip()
    if not question_text:
        return jsonify({"detail": "Question cannot be empty."}), 400

    try:
        response_data = ask(question_text, return_structured=True)
        return jsonify(response_data), 200
    except Exception as error_exception:
        return jsonify({"detail": f"RAG Error: {str(error_exception)}"}), 500


# ==========================================
# 3. FILE MANAGEMENT & MEDIA SERVING
# ==========================================

@app.route("/api/files", methods=["GET"])
def list_files():
    """Lists all indexed files stored in LanceDB processed_files hash table."""
    try:
        htable = get_hash_table()
        df = htable.to_pandas()
        if df.empty:
            return jsonify({"files": []}), 200
        
        records = df.to_dict(orient="records")
        # Format response
        files = []
        for r in records:
            path_str = r.get("path", "")
            filename = os.path.basename(path_str) if path_str else "Unknown"
            
            # Detect file type
            ext = os.path.splitext(filename)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                ftype = "image"
            elif ext in [".mp3", ".wav", ".mpeg", ".aac", ".flac"]:
                ftype = "audio"
            elif ext in [".mp4", ".mkv", ".avi", ".mov"]:
                ftype = "video"
            elif ext in [".pdf"]:
                ftype = "pdf"
            elif ext in [".docx", ".doc"]:
                ftype = "docx"
            elif ext in [".xlsx", ".xls"]:
                ftype = "xlsx"
            else:
                ftype = "document"

            files.append({
                "file_hash": r.get("file_hash"),
                "filename": filename,
                "path": path_str,
                "file_type": ftype,
                "created_at": r.get("created_at")
            })

        return jsonify({"files": files}), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


@app.route("/api/media/<path:file_path>", methods=["GET"])
def serve_media(file_path: str):
    """
    Serves images, audio, video, or documents directly to the UI.
    """
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = BASE_DIR / file_path

    if not path_obj.exists():
        # Try finding in watched_folder or root
        alt_path = WATCHED_FOLDER / os.path.basename(file_path)
        if alt_path.exists():
            path_obj = alt_path
        else:
            return jsonify({"detail": f"File not found: {file_path}"}), 404

    return send_file(str(path_obj))


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Uploads a file to watched_folder and indexes it immediately."""
    try:
        uploaded_file = request.files.get("file")
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({"detail": "No file uploaded."}), 400

        filename = uploaded_file.filename
        dest_path = WATCHED_FOLDER / filename
        uploaded_file.save(str(dest_path))

        # Trigger immediate extraction & indexing
        result = extract_file(str(dest_path))
        return jsonify({
            "status": "SUCCESS",
            "filename": filename,
            "path": str(dest_path),
            "result": result
        }), 200
    except Exception as error_exception:
        return jsonify({"detail": f"Upload processing failed: {str(error_exception)}"}), 500


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
        uploaded_file = request.files.get("file")

        target_path = image_path
        if uploaded_file and uploaded_file.filename:
            dest_path = WATCHED_FOLDER / uploaded_file.filename
            uploaded_file.save(str(dest_path))
            target_path = str(dest_path)

        if not target_path or not person_name:
            return jsonify({"detail": "Either image_path or file must be provided along with person_name."}), 400

        result = register_face(person_name=person_name, image_path=target_path)
        return jsonify(result), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


# ==========================================
# 5. AUDIO & VIDEO SPECIFIC APIs
# ==========================================

@app.route("/api/audio", methods=["GET"])
def get_audio_files():
    """Returns all processed audio files with their transcript snippets."""
    try:
        doc_table = get_table()
        df = doc_table.to_pandas()
        if df.empty:
            return jsonify({"audio": []}), 200

        audio_df = df[df["file_type"] == "audio"]
        records = audio_df.to_dict(orient="records")

        # Group by audio path
        grouped = {}
        for r in records:
            p = r["path"]
            if p not in grouped:
                grouped[p] = {
                    "filename": os.path.basename(p),
                    "path": p,
                    "transcripts": []
                }
            grouped[p]["transcripts"].append(r["text"])

        return jsonify({"audio": list(grouped.values())}), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


@app.route("/api/video", methods=["GET"])
def get_video_files():
    """Returns processed video files with keyframe & audio transcript metadata."""
    try:
        doc_table = get_table()
        df = doc_table.to_pandas()
        if df.empty:
            return jsonify({"videos": []}), 200

        video_df = df[df["file_type"].isin(["video", "video_frame"])]
        records = video_df.to_dict(orient="records")

        grouped = {}
        for r in records:
            p = r["path"]
            if p not in grouped:
                grouped[p] = {
                    "filename": os.path.basename(p),
                    "path": p,
                    "chunks": []
                }
            grouped[p]["chunks"].append({
                "chunk_id": r["chunk_id"],
                "text": r["text"]
            })

        return jsonify({"videos": list(grouped.values())}), 200
    except Exception as error_exception:
        return jsonify({"detail": str(error_exception)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
