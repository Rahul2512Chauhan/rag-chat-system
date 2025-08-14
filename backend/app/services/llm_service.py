# backend/app/services/llm_service.py
import os
from typing import List, Optional, Callable, Dict, Any, cast

from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from pydantic import SecretStr
from app.services.vector_service import get_vector_store

load_dotenv()


def get_llm() -> ChatGroq:
    """
    Return a LangChain-compatible LLM client (Groq).
    Requires GROQ_API_KEY in environment or .env.
    Optional: GROQ_MODEL to override the default model.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

    api_key_str: str = cast(str, api_key)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ChatGroq expects api_key (string). Some wrappers accept SecretStr; using string is fine.
    return ChatGroq(
        model=model,
        temperature=0,
        api_key=SecretStr(api_key_str)
    )


def get_rag_chain(
    memory: ConversationBufferMemory,
    k: int = 5,
    store_path: Optional[str] = None
) -> ConversationalRetrievalChain:
    """
    Build a conversational retrieval chain using the LLM, retriever and memory.
    NOTE: We intentionally avoid using ChatPromptTemplate + MessagesPlaceholder here
    to prevent type-mismatch issues with chat_history variable injection.
    """
    vectordb = get_vector_store(store_path) if store_path else get_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # Let the default combine / prompt behavior run; the memory will be attached to the chain.
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    return rag_chain


def get_deep_research_chain(k: int = 10, store_path: Optional[str] = None) -> Callable[[str], Dict[str, Any]]:
    """
    Returns a function that performs deep research (retrieve -> summarize -> answer).
    """
    vectordb = get_vector_store(store_path) if store_path else get_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = get_llm()
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False)

    def deep_research(query: str) -> Dict[str, Any]:
        docs = retriever.get_relevant_documents(query)
        summary_result = summarize_chain.run(docs)

        prompt = f"You are an expert assistant. Use this summary to answer:\n\n{summary_result}\n\nQuestion: {query}\nAnswer:"
        # ChatGroq may return an object; try to get content attribute or string
        answer = llm.invoke(prompt)
        answer_text = getattr(answer, "content", str(answer))

        sources: List[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            slide = doc.metadata.get("slide")
            loc = f"Page {page}" if page else (f"Slide {slide}" if slide else "")
            src_str = f"{source} {loc}".strip()
            if src_str and src_str not in sources:
                sources.append(src_str)

        return {"answer": answer_text, "sources": sources}

    return deep_research
