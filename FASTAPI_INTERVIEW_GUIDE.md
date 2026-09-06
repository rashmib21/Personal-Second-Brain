# FastAPI Interview Mastery & Architecture Guide
### A Complete Guide to Cracking FastAPI System Design & Coding Interviews

This guide covers everything you need to master **FastAPI** for technical interviews—ranging from low-level architecture to real-world code patterns from your **Personal Second Brain** application (`app/server.py`).

---

## 📑 Table of Contents
1. [Core Architecture & Technical Stack](#1-core-architecture--technical-stack)
2. [WSGI vs. ASGI (Why FastAPI is Ultra Fast)](#2-wsgi-vs-asgi-why-fastapi-is-ultra-fast)
3. [Async/Await & Concurrency in FastAPI](#3-asyncawait--concurrency-in-fastapi)
4. [FastAPI Core Components with Real Project Code](#4-fastapi-core-components-with-real-project-code)
5. [Production Deployment & Scaling](#5-production-deployment--scaling)
6. [Top 15 FastAPI Interview Questions & Winning Answers](#6-top-15-fastapi-interview-questions--winning-answers)
7. [How to Answer "How did you use FastAPI in your project?" in Interviews](#7-how-to-answer-how-did-you-use-fastapi-in-your-project-in-interviews)

---

## 1. Core Architecture & Technical Stack

FastAPI is a modern, high-performance Python web framework built on top of two core foundational libraries:

```
                  ┌─────────────────────────────────────────┐
                  │                FastAPI                  │
                  └──────────────────┬──────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
┌────────────▼──────────────┐                 ┌──────────────▼─────────────┐
│    Starlette (Web / ASGI) │                 │ Pydantic (Data Validation) │
└───────────────────────────┘                 └────────────────────────────┘
```

1. **Starlette**: Handles web routing, ASGI execution, HTTP requests/responses, WebSockets, background tasks, and CORS middleware.
2. **Pydantic**: Handles data validation, type hints, serialization (Python objects -> JSON), and deserialization using pure Python type annotations.
3. **OpenAPI / Swagger UI**: Generates interactive API documentation at `/docs` and `/redoc` automatically from your Python type hints.

---

## 2. WSGI vs. ASGI (Why FastAPI is Ultra Fast)

Interviews frequently compare traditional frameworks (Flask, Django) with FastAPI.

| Feature | Traditional Frameworks (Flask / Django) | FastAPI |
| :--- | :--- | :--- |
| **Server Standard** | **WSGI** (Web Server Gateway Interface) | **ASGI** (Asynchronous Server Gateway Interface) |
| **Concurrency Model** | Synchronous, Blocking, 1 Thread per Request | Asynchronous, Non-blocking Event Loop |
| **Async Support** | Limited / Emulated | Native `async` / `await` |
| **Performance** | ~2,000 - 5,000 req/sec | ~20,000 - 30,000 req/sec (comparable to Node.js & Go) |
| **Data Validation** | Manual (marshmallow, wtforms) | Automatic via Pydantic |

### Key Interview Concept:
> **WSGI** is synchronous—if a request is waiting for a database query or ML model inference, that server thread is completely blocked.
> **ASGI** is asynchronous—when a request waits for an I/O operation (e.g. database query, disk read, network request), the single-threaded event loop immediately pauses that request (`await`) and handles hundreds of other incoming requests in the meantime.

---

## 3. Async/Await & Concurrency in FastAPI

### When should you use `async def` vs normal `def`?

1. **Use `async def`** when your function contains I/O-bound operations using `await` (e.g., async database calls, fetching external APIs with `httpx`, reading file streams):
   ```python
   @app.post("/api/chat")
   async def chat_endpoint(payload: dict):
       # Non-blocking async operation
       response = await async_rag_search(payload["question"])
       return response
   ```

2. **Use normal `def`** when calling standard synchronous Python libraries (like PyTorch, OpenCV, LanceDB, or blocking CPU operations):
   ```python
   @app.get("/api/stats")
   def get_system_stats():
       # FastAPI automatically runs normal def functions in an external ThreadPool
       t_chunks = total_chunks()  # Synchronous LanceDB call
       return {"chunks": t_chunks}
   ```

> 💡 **Interview Pro-Tip**: If you write `async def` BUT call a blocking synchronous library inside it *without* `await`, you will **block the main event loop**! If your code is synchronous, write a normal `def` function—FastAPI will safely run it in a separate worker threadpool.

---

## 4. FastAPI Core Components with Real Project Code

Here are the 6 essential patterns implemented in your **Personal Second Brain** (`app/server.py`):

### 1. App Initialization & CORS Middleware
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Personal Second Brain API",
    version="2.0.0"
)

# CORS Allows your frontend JavaScript to call API endpoints from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Serving Static Single Page Applications (SPA)
```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# Mount /static route for CSS & JS files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def read_index():
    # Return index.html on root route
    return FileResponse(str(FRONTEND_DIR / "index.html"))
```

### 3. Wildcard Path Parameters for Media Serving
```python
# {file_path:path} captures multi-directory paths like "watched_folder/images/photo.jpg"
@app.get("/api/media/{file_path:path}")
async def serve_media(file_path: str):
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # FileResponse handles MIME types & HTTP byte-range streaming for video/audio
    return FileResponse(str(path_obj))
```

### 4. Handling File Uploads (`UploadFile` vs `bytes`)
```python
from fastapi import UploadFile, File, Form

@app.post("/api/faces/register")
async def register_face_endpoint(
    person_name: str = Form(...),          # Multipart form data
    file: UploadFile = File(...)           # Uploaded binary file stream
):
    # UploadFile streams to disk/spool rather than loading entire file into RAM
    dest_path = WATCHED_FOLDER / file.filename
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    result = register_face(person_name, str(dest_path))
    return result
```

### 5. Pydantic Request Validation Models
```python
from pydantic import BaseModel, Field

class ChatQueryRequest(BaseModel):
    question: str = Field(..., example="Show me the picture of Rashmi")
    top_k: int = Field(default=5, ge=1, le=50)

@app.post("/api/chat")
async def chat_endpoint(payload: ChatQueryRequest):
    # Payload is guaranteed to have valid question string & top_k integer
    res = ask(payload.question, return_structured=True)
    return res
```

---

## 5. Production Deployment & Scaling

To run FastAPI in production, you use an **ASGI Server**:

```bash
# Development (with hot-reload):
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

# Production (Multi-Worker Execution):
gunicorn app.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

* **Uvicorn**: Lightning-fast ASGI server built on `uvloop` (Python binding for `libuv`, the C library behind Node.js).
* **Gunicorn + Uvicorn Workers**: Gunicorn acts as a process manager running multiple Uvicorn worker instances across CPU cores.

---

## 6. Top 15 FastAPI Interview Questions & Winning Answers

### Q1: What is FastAPI and why would you choose it over Flask or Django?
> **Answer**: FastAPI is a modern Python ASGI framework built on Starlette and Pydantic. I choose it because:
> 1. **Extreme Performance**: On par with Node.js and Go due to ASGI and `uvloop`.
> 2. **Automatic Type Validation**: Uses Pydantic to validate requests/responses automatically.
> 3. **Auto-Generated Docs**: Provides instant Swagger UI (`/docs`) and ReDoc (`/redoc`).
> 4. **Developer Efficiency**: Reduces boilerplate code by 40%.

### Q2: What is the difference between WSGI and ASGI?
> **Answer**: WSGI (Flask/Django) is a synchronous server standard where each request blocks a thread until execution finishes. ASGI (FastAPI) is an asynchronous standard that supports Python's event loop, allowing non-blocking I/O, WebSockets, and high concurrency.

### Q3: What happens if you run a blocking synchronous call inside an `async def` endpoint?
> **Answer**: It blocks the main event loop, preventing all other concurrent async requests from being processed! If an endpoint must run synchronous blocking code (like heavy ML inference or file I/O), it should be declared as a standard `def` function so FastAPI offloads it to a background threadpool (`anyio.to_thread.run_sync`).

### Q4: How does FastAPI handle data validation?
> **Answer**: FastAPI relies on Pydantic. When a client sends a JSON request, FastAPI checks the fields against Pydantic models or Python type hints. If valid, it passes typed objects to the function; if invalid, it automatically returns an HTTP 422 Unprocessable Entity error with exact field validation details.

### Q5: What is `UploadFile` and why is it better than receiving raw `bytes` for file uploads?
> **Answer**: `bytes` reads the entire uploaded file directly into RAM, which will crash the server if a user uploads a 2 GB video. `UploadFile` uses a spooled file buffer—storing data in memory up to a small threshold (1 MB) and then streaming larger chunks directly to disk, preserving system memory.

### Q6: How do Dependencies (`Depends`) work in FastAPI?
> **Answer**: FastAPI has a built-in Dependency Injection system using `Depends()`. It allows you to share database connections, authentication logic, rate limiters, or request validators across endpoints cleanly without code duplication.

### Q7: How do you handle CORS in FastAPI?
> **Answer**: Using `CORSMiddleware`. You add the middleware to the app instance and specify allowed origins, methods (GET, POST), and headers so browser security policies allow frontend clients on different ports/domains to access the API.

### Q8: How do you handle custom error responses in FastAPI?
> **Answer**: By raising `HTTPException(status_code=404, detail="Item not found")` or creating custom exception handlers using `@app.exception_handler(CustomException)`.

### Q9: How do you serve static files and media in FastAPI?
> **Answer**: Using `app.mount("/static", StaticFiles(directory="..."))` for assets, and `FileResponse(filepath)` for returning media streams (images, PDFs, MP3s, MP4s) directly to the client.

### Q10: How do you stream large media files (like audio or video) in FastAPI?
> **Answer**: Using `FileResponse` or `StreamingResponse`. `FileResponse` supports HTTP range requests natively, allowing HTML5 `<audio>` and `<video>` players to seek forwards and backwards without loading the full file at once.

### Q11: How do you structure a large production FastAPI application?
> **Answer**: Using `APIRouter` to split routes into modular domains (e.g. `routes/auth.py`, `routes/chat.py`, `routes/files.py`), registered on the main app with `app.include_router(chat_router)`.

### Q12: What is the difference between `BackgroundTasks` in FastAPI and Celery?
> **Answer**: FastAPI `BackgroundTasks` run in the same process after sending an HTTP response—ideal for quick tasks like sending an email. Celery is a distributed task queue running in separate worker processes with Redis/RabbitMQ brokers—essential for heavy tasks like video processing, ML embedding generation, or ASR transcription.

### Q13: How do you handle authentication in FastAPI?
> **Answer**: Using `fastapi.security` (like `OAuth2PasswordBearer`) combined with `python-jose` for JWT (JSON Web Tokens) verification passed via the `Authorization: Bearer <token>` header inside a `Depends()` security handler.

### Q14: What is Pydantic's `BaseModel` vs dataclasses?
> **Answer**: Standard dataclasses only store data types without runtime enforcement. Pydantic `BaseModel` performs actual runtime validation, type coercion (e.g. converting `"123"` string to integer `123`), custom validators (`@field_validator`), and JSON schema generation.

### Q15: How do you monitor and document FastAPI endpoints?
> **Answer**: OpenAPI is auto-generated at `/docs` (Swagger) and `/redoc`. Production monitoring is done using Prometheus metrics middleware (`starlette-prometheus`) or APM tools like Sentry / Datadog.

---

## 7. How to Answer "How did you use FastAPI in your project?" in Interviews

When interviewers ask about your project, use this structured STAR response:

> **"In my Personal Second Brain project, I built a high-performance REST backend using FastAPI to connect our single-page frontend to our multimodal AI RAG pipeline."**
>
> 1. **Architecture**: I used FastAPI built on Starlette and Pydantic. It exposes endpoints for RAG text queries (`/api/chat`), file listing (`/api/files`), media streaming (`/api/media`), and face memory registration (`/api/faces/register`).
> 2. **Performance & Concurrency**: I designed endpoints to be non-blocking. For media streaming (audio & video), I used `FileResponse` which supports HTTP byte-range streaming, allowing HTML5 players to seek smoothly without memory spikes.
> 3. **Data Integrity & Validation**: I used Pydantic models for incoming JSON payloads and handles file uploads with `UploadFile` to stream large media without exhausting RAM.
> 4. **Integration**: FastAPI served as the orchestrator—routing vector queries to LanceDB, vision requests to Qwen2.5-VL, face searches to FaceNet 512-D vectors, and heavy background jobs to Celery.
