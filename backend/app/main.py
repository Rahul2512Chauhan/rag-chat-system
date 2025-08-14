# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, chat
from pathlib import Path
import os

from dotenv import load_dotenv   
load_dotenv()

# --- Ensure vector store directory exists ---
BASE_DIR = Path(__file__).resolve().parent.parent  # points to backend/
CHROMA_DIR = BASE_DIR / "chroma_store"
CHROMA_DIR.mkdir(parents=True , exist_ok=True)  # creates if not exists

# Debug prints to confirm paths
print("Backend base directory:", BASE_DIR)
print("Chroma vector store path:", CHROMA_DIR)
print("Is 'chroma_store' writable?", os.access(CHROMA_DIR, os.W_OK))

# --- Initialize FastAPI app ---
app = FastAPI(
    title="RAG Chat Backend",
    description="Backend API for Retrieval-Augmented Generation chat system",
    version="1.0.0"
)

# --- CORS middleware for frontend (Streamlit) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include API routers ---
app.include_router(upload.router, prefix="/api", tags=["Document Upload"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])

# --- Root endpoint for health check ---
@app.get("/")
async def root():
    return {"message": "RAG Chat Backend is running"}
