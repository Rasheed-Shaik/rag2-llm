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
from rag_methods import stream_llm_response, stream_llm_rag_response, load_doc_to_db, load_url_to_db, initialize_documents

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
    initialize_documents()  # Load persisted documents on startup

# Rest of the code remains the same...