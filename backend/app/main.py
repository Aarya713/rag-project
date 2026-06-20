from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
import os
from typing import List

from .models import PDFUploadResponse, QuestionRequest, AnswerResponse
from .rag_engine import RAGEngine

app = FastAPI(title="PDF RAG API", version="1.0")

# CORS – allow React frontend (if you still use it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()

# ---------- API endpoints ----------
@app.post("/upload", response_model=PDFUploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded.")

    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        dst = os.path.join(temp_dir, file.filename)
        with open(dst, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_paths.append(dst)

    if not saved_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(400, "No valid PDF files uploaded.")

    try:
        num_chunks, num_files = engine.process_pdfs(saved_paths)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(500, f"Processing failed: {str(e)}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    return PDFUploadResponse(
        status="success",
        num_chunks=num_chunks,
        num_files=num_files
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    try:
        return engine.ask(req.question)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error generating answer: {str(e)}")

@app.get("/status")
async def status():
    ready = engine.vectorstore is not None
    return {"ready": ready, "chunks": len(engine.chunks)}

# ---------- Serve frontend static files ----------
# Go up two levels from backend/app/main.py -> project root, then into frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")