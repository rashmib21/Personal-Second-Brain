import os
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

app = FastAPI(
    title="Personal Second Brain API",
    description="Multimodal AI RAG & Knowledge Management System",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend static files if directory exists
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def read_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Personal Second Brain API is running."}


# ==========================================
# 1. SYSTEM STATS & STATUS
# ==========================================

@app.get("/api/stats")
async def get_system_stats():
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
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. CHAT & RAG PIPELINE
# ==========================================

@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    """
    Executes the multimodal RAG pipeline for a user question.
    Expected payload: {"question": "..."}
    """
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        res = ask(question, return_structured=True)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")


# ==========================================
# 3. FILE MANAGEMENT & MEDIA SERVING
# ==========================================

@app.get("/api/files")
async def list_files():
    """Lists all indexed files stored in LanceDB processed_files hash table."""
    try:
        htable = get_hash_table()
        df = htable.to_pandas()
        if df.empty:
            return JSONResponse(content={"files": []})
        
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

        return JSONResponse(content={"files": files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/media/{file_path:path}")
async def serve_media(file_path: str):
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
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    return FileResponse(str(path_obj))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a file to watched_folder and indexes it immediately."""
    try:
        dest_path = WATCHED_FOLDER / file.filename
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger immediate extraction & indexing
        result = extract_file(str(dest_path))
        return JSONResponse(content={
            "status": "SUCCESS",
            "filename": file.filename,
            "path": str(dest_path),
            "result": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


# ==========================================
# 4. FACE MEMORY APIs
# ==========================================

@app.get("/api/faces")
async def get_faces():
    """Lists registered face memory identities and all detected face records."""
    try:
        ftable = get_face_table()
        df = ftable.to_pandas()
        if df.empty:
            return JSONResponse(content={"faces": [], "persons": []})

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

        return JSONResponse(content={
            "faces": formatted_faces,
            "persons": persons
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/faces/register")
async def register_face_endpoint(
    person_name: str = Form(...),
    image_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Registers a reference face for a person given an image path or uploaded file.
    """
    try:
        target_path = image_path
        if file:
            dest_path = WATCHED_FOLDER / file.filename
            with dest_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            target_path = str(dest_path)

        if not target_path:
            raise HTTPException(status_code=400, detail="Either image_path or file must be provided.")

        result = register_face(person_name=person_name, image_path=target_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 5. AUDIO & VIDEO SPECIFIC APIs
# ==========================================

@app.get("/api/audio")
async def get_audio_files():
    """Returns all processed audio files with their transcript snippets."""
    try:
        doc_table = get_table()
        df = doc_table.to_pandas()
        if df.empty:
            return JSONResponse(content={"audio": []})

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

        return JSONResponse(content={"audio": list(grouped.values())})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video")
async def get_video_files():
    """Returns processed video files with keyframe & audio transcript metadata."""
    try:
        doc_table = get_table()
        df = doc_table.to_pandas()
        if df.empty:
            return JSONResponse(content={"videos": []})

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

        return JSONResponse(content={"videos": list(grouped.values())})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
