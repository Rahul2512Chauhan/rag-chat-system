# backend/app/api/chat.py
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
from langchain.memory import ConversationBufferMemory
import json

from app.services.llm_service import get_rag_chain, get_deep_research_chain
from app.services.vector_service import resolve_path

router = APIRouter()

# Simple in-memory session store (replace with DB/redis in production)
session_memories: Dict[str, ConversationBufferMemory] = {}


@router.get("/chat")
async def chat_endpoint(
    session_id: str = Query(..., description="Unique session identifier"),
    query: str = Query(..., description="User's input question"),
    mode: str = Query("standard", description="'standard' for RAG, 'deep' for research mode"),
    chroma_dir: Optional[str] = Query(None, description="Path to vector store directory"),
    k: int = Query(5, description="Number of documents to retrieve"),
    chat_history: Optional[str] = Query(None, description="Frontend JSON string of chat history")
):
    store_path = resolve_path(chroma_dir)

    try:
        # --- Deep Research Mode ---
        if mode.lower() == "deep":
            deep_research_fn = get_deep_research_chain(k=k, store_path=store_path)
            result = deep_research_fn(query)
            return JSONResponse(content=result, status_code=200)

        # --- Standard RAG Mode ---
        if session_id not in session_memories:
            session_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"  # ✅ Fix: tell memory what to store
            )

        memory = session_memories[session_id]

        # If front-end sent prior history, reconstruct it into the memory properly
        if chat_history:
            try:
                parsed_history = json.loads(chat_history)
                if not isinstance(parsed_history, list):
                    raise ValueError("chat_history must be a JSON list of {question, answer} objects")

                for item in parsed_history:
                    q_text = item.get("question", "")
                    a_text = item.get("answer", "")
                    if q_text:
                        memory.save_context({"question": q_text}, {"answer": a_text or ""})

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid chat_history: {e}")

        # Build and run chain
        rag_chain = get_rag_chain(memory=memory, k=k, store_path=store_path)
        result = rag_chain.invoke({"question": query})

        # Extract answer and sources
        answer_text = result.get("answer") or result.get("result", "")

        sources: List[str] = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            slide = doc.metadata.get("slide")
            loc = f"Page {page}" if page else (f"Slide {slide}" if slide else "")
            src_str = f"{src} {loc}".strip()
            if src_str and src_str not in sources:
                sources.append(src_str)

        return JSONResponse(content={"answer": answer_text, "sources": sources}, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
