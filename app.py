import streamlit as st
import os
import uuid
# SQLite fix for Streamlit Cloud
import platform
if platform.system() != "Windows":
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import stream_llm_response, stream_llm_rag_response, load_doc_to_db, load_url_to_db

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Chat App",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session states
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.rag_sources = []
    st.session_state.vector_db = None
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# Page header
st.markdown("""<h2 style="text-align: center;">📚 RAG-Enabled Chat Assistant 🤖</h2>""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # API Key Management
    with st.expander("🔐 API Keys", expanded=True):
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            key="google_api_key"
        )

    # Model Selection and Chat Controls
    model = "google/gemini-pro"  # Using a stable model
    st.session_state.use_rag = st.toggle(
        "Enable RAG",
        value=st.session_state.vector_db is not None,
        disabled=st.session_state.vector_db is None
    )
    
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages.clear()
        st.rerun()

    # RAG Document Management
    st.header("📚 Knowledge Base")
    st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs"
    )

    st.text_input(
        "Add Website URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url"
    )

    with st.expander(f"📂 Loaded Sources ({len(st.session_state.rag_sources)})"):
        st.write(st.session_state.rag_sources)

# Main chat interface
if not google_api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to continue.")
else:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=model.split("/")[-1],
        google_api_key=google_api_key,
        temperature=0.7,
        streaming=True
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and response
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages
            ]
            
            if st.session_state.use_rag:
                st.write_stream(stream_llm_rag_response(llm, messages))
            else:
                st.write_stream(stream_llm_response(llm, messages))