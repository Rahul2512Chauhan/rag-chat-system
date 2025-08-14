# backend/app/api/upload.py
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import shutil

from app.services.document_service import load_documents
from app.services.vector_service import add_documents, resolve_path, reset_vector_store

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
UPLOAD_DIR = BASE_DIR / "uploaded_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    chroma_dir: Optional[str] = Form(None)
):
    if not files:
        return JSONResponse(
            content={"error": "No files were uploaded."},
            status_code=400
        )

    store_path = resolve_path(chroma_dir)

    # Reset once for the whole batch
    reset_vector_store(store_path)

    saved_files: List[str] = []
    extracted_count = 0
    failed: List[str] = []

    for file in files:
        if not file.filename:
            failed.append("(unnamed file)")
            continue

        file_path = UPLOAD_DIR / file.filename

        # Save uploaded file
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_files.append(file.filename)
        except Exception as e:
            failed.append(f"{file.filename} (save error: {e})")
            continue

        # Extract text and add to vector DB
        try:
            docs = load_documents(str(file_path))
            if docs:
                extracted_count += len(docs)
                add_documents(docs, store_path=store_path, reset=False)
        except Exception as e:
            failed.append(f"{file.filename} (process error: {e})")

    status = 200 if extracted_count > 0 else 500
    return JSONResponse(
        content={
            "message": "Ingestion completed",
            "files_saved": saved_files,
            "documents_indexed": extracted_count,
            "failed": failed,
            "chroma_dir_used": store_path
        },
        status_code=status
    )
