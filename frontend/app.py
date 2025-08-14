import streamlit as st
import requests
import uuid
import os
import json

# --- Absolute path to backend vector store ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../backend/chroma_store"))
os.makedirs(CHROMA_DIR, exist_ok=True)

BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Initialize session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "Standard Chat"
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False


def upload_files():
    st.markdown("<h3 style='color:#1D4ED8; font-weight:700;'>📂 Upload Documents</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag and drop files or click to browse (PDF, PPTX, DOCX, TXT, CSV, XLSX)",
        type=["pdf", "pptx", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Max file size: 200MB each."
    )

    if uploaded_files:
        files = [("files", (f.name, f, f.type)) for f in uploaded_files]
        with st.spinner("Uploading and processing files..."):
            try:
                response = requests.post(
                    f"{BASE_URL}/upload",
                    files=files,
                    data={"chroma_dir": CHROMA_DIR}
                )
                if response.status_code == 200:
                    st.success("✅ Files uploaded and indexed successfully!")
                    st.session_state.files_uploaded = True
                    st.session_state.chat_history = []
                else:
                    st.error(f"Upload failed: {response.text}")
                    st.session_state.files_uploaded = False
            except Exception as e:
                st.error(f"Upload error: {e}")
                st.session_state.files_uploaded = False
    elif not st.session_state.files_uploaded:
        st.info("Please upload at least one document to start chatting.")


def chat_interface():
    st.markdown("---")
    st.markdown("<h2 style='color:#58A6FF; font-weight:800;'>💬 Chat with Your Documents</h2>", unsafe_allow_html=True)

    # Mode selection
    mode = st.selectbox(
        "Select Mode",
        options=["Standard Chat", "Deep Research"],
        index=0 if st.session_state.mode == "Standard Chat" else 1,
        key="mode_selector"
    )
    st.session_state.mode = mode

    # Query form
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about your documents:",
            key="input_query",
            placeholder="Type your question here..."
        )
        ask_clicked = st.form_submit_button("Ask")

        if ask_clicked:
            if not query.strip():
                st.warning("⚠️ Please enter a question before asking.")
            else:
                with st.spinner("Fetching answer..."):
                    try:
                        limited_history = st.session_state.chat_history[-10:]

                        params = {
                            "query": query,
                            "session_id": st.session_state.session_id,
                            "mode": "deep" if st.session_state.mode == "Deep Research" else "standard",
                            "chroma_dir": CHROMA_DIR,
                            # Always send JSON string
                            "chat_history": json.dumps([
                                {"question": c["question"], "answer": c["answer"]}
                                for c in limited_history
                            ])
                        }

                        response = requests.get(f"{BASE_URL}/chat", params=params)
                        if response.status_code == 200:
                            data = response.json()
                            answer = data.get("answer", "No answer returned.")
                            sources = data.get("sources", [])
                            st.session_state.chat_history.append({
                                "question": query,
                                "answer": answer,
                                "sources": sources
                            })
                        else:
                            st.error(f"Server error: {response.text}")
                    except Exception as e:
                        st.error(f"Request error: {e}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("<h3 style='color:#58A6FF; font-weight:700;'>📖 Chat History</h3>", unsafe_allow_html=True)
        for chat in reversed(st.session_state.chat_history):
            st.markdown(
                f"""
                <div style="
                    background-color:#0D1117;  
                    padding:20px;
                    border-radius:12px;
                    margin-bottom:20px;
                    border: 1.5px solid #58A6FF;
                ">
                    <p style="margin:0; font-weight:700; font-size:17px; color:#C9D1D9;">Q: {chat['question']}</p>
                    <p style="margin:12px 0 0 0; font-size:16px; color:#F0F6FC;">A: {chat['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if chat["sources"]:
                st.markdown("<p style='font-weight:600; color:#58A6FF; margin-top:-14px;'>Sources:</p>", unsafe_allow_html=True)
                for src in chat["sources"]:
                    st.markdown(f"- <span style='color:#58A6FF'>{src}</span>")
                st.markdown("---")


def main():
    st.markdown(
        "<h1 style='color:#58A6FF; font-weight:900;'>🤖 Retrieval-Augmented Generation (RAG) Chat System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:16px; color:#8B949E;'>Upload your documents to get instant, context-aware answers. "
        "Switch to <b>Deep Research</b> mode for detailed insights.</p>",
        unsafe_allow_html=True
    )

    upload_files()

    if st.session_state.files_uploaded:
        chat_interface()
    else:
        st.info("Upload files above to start chatting.")


if __name__ == "__main__":
    main()
