# backend/app/services/vector_service.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List

from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from app.services.embedding_service import get_embeddings_model

# Base: backend/
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_DIR = str(BASE_DIR / "chroma_store")


def resolve_path(store_path: Optional[str] = None) -> str:
    """
    Resolve a vector-store directory path to an absolute path under backend/.
    - If store_path is None -> use DEFAULT_VECTOR_DIR.
    - If store_path is relative -> resolve relative to backend/.
    - Ensures the directory exists.
    """
    if not store_path or not str(store_path).strip():
        abs_path = Path(DEFAULT_VECTOR_DIR)
    else:
        p = Path(store_path)
        abs_path = p if p.is_absolute() else (BASE_DIR / p)

    abs_path.mkdir(parents=True, exist_ok=True)
    return str(abs_path)


def reset_vector_store(store_path: Optional[str] = None) -> None:
    """
    Delete persisted vector store directory to clear previous embeddings.
    """
    abs_path = Path(resolve_path(store_path))
    if abs_path.exists():
        shutil.rmtree(abs_path)
    abs_path.mkdir(parents=True, exist_ok=True)


def get_vector_store(store_path: Optional[str] = None) -> Chroma:
    """
    Return a Chroma vector store instance at the given path.
    """
    abs_path = resolve_path(store_path)
    embeddings = get_embeddings_model()
    vectordb = Chroma(
        persist_directory=abs_path,
        embedding_function=embeddings,
    )
    return vectordb


def add_documents(
    docs: List[Document],
    store_path: Optional[str] = None,
    reset: bool = False
) -> int:
    """
    Add new documents to the vector store.
    - reset=True clears old data first
    - reset=False appends to existing store
    Always persists after adding.
    """
    if reset:
        reset_vector_store(store_path)

    vectordb = get_vector_store(store_path)
    if docs:
        vectordb.add_documents(docs)
        # Ensure data is flushed to disk
        vectordb.persist()
    return len(docs)


def retrieve_documents(
    query: str,
    k: int = 5,
    store_path: Optional[str] = None
):
    """
    Retrieve top-k relevant documents for a query.
    """
    vectordb = get_vector_store(store_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs
