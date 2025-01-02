# app.py
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
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I assist you today?")
    ]
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Page header
st.markdown("""<h2 style="text-align: center;">📚 RAG-Enabled Chat Assistant 🤖</h2>""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # API Key Management - Check secrets first, then environment, then input
    google_api_key = st.secrets.get("google_api_key", "") if hasattr(st, "secrets") else ""

    # Only show API input if no key in secrets
    if not google_api_key:
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            key="google_api_key"
        )

    # Model Selection and Chat Controls
    model = "google/gemini-1.5-flash-latest"  # Using a stable model
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False  # Default to False

    st.write(f"Sidebar: Before RAG Toggle - st.session_state.vector_db is {st.session_state.vector_db}")
    st.write(f"Sidebar: Before RAG Toggle - st.session_state.use_rag is {st.session_state.use_rag}")

    st.session_state.use_rag = st.toggle(
        "Enable RAG",
        value=st.session_state.vector_db is not None,
        disabled=st.session_state.vector_db is None
    )

    st.write(f"Sidebar: After RAG Toggle - st.session_state.vector_db is {st.session_state.vector_db}")
    st.write(f"Sidebar: After RAG Toggle - st.session_state.use_rag is {st.session_state.use_rag}")

    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I assist you today?")
        ]
        st.rerun()

    # RAG Document Management
    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="rag_docs"
    )
    if uploaded_files:
        st.write("app.py: Uploaded files detected, calling load_doc_to_db")
        load_doc_to_db(uploaded_files)
        st.write(f"app.py: After load_doc_to_db - st.session_state.vector_db is {st.session_state.vector_db}")

    url_input = st.text_input(
        "Add Website URL",
        placeholder="https://example.com",
        key="rag_url"
    )
    if url_input:
        st.write("app.py: URL input detected, calling load_url_to_db")
        load_url_to_db(url_input)
        st.write(f"app.py: After load_url_to_db - st.session_state.vector_db is {st.session_state.vector_db}")

    with st.expander(f"📂 Loaded Sources ({len(st.session_state.rag_sources)})"):
        st.write(st.session_state.rag_sources)
        if "documents_loaded" not in st.session_state:
            st.write("Expander: documents_loaded not in session_state")
            if "session_id" in st.session_state:  # Ensure session_id is initialized
                st.write("Expander: session_id found, calling initialize_documents")
                initialize_documents()
                st.session_state.documents_loaded = True
                st.write("Expander: documents_loaded set to True")
        else:
            st.write("Expander: documents_loaded is in session_state")

# Main chat interface
if not google_api_key:
    st.warning("⚠️ No Google API Key found. Please add it to your Streamlit secrets or enter it in the sidebar.")
else:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=model.split("/")[-1],
        google_api_key=google_api_key,
        temperature=0.7,
        streaming=True
    )

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        st.write(f"Debugging message[{i}]: {message}, type: {type(message)}")
        if not hasattr(message, 'role'):
            st.error(f"Error: message[{i}] does not have a 'role' attribute. Type: {type(message)}, Attributes: {message.__dict__ if hasattr(message, '__dict__') else dir(message)}")
        try:
            with st.chat_message(message.role):
                st.markdown(message.content)
        except Exception as e:
            st.error(f"Error displaying message[{i}]: {e}")

    # Chat input and response
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write(f"Chat Loop: st.session_state.use_rag is {st.session_state.use_rag}")
            st.write(f"Chat Loop: st.session_state.vector_db is {st.session_state.vector_db}")
            if st.session_state.use_rag and st.session_state.vector_db is not None:
                st.write("Chat Loop: Executing RAG Response")
                st.write_stream(stream_llm_rag_response(llm, st.session_state.messages))
            else:
                st.write("Chat Loop: Executing Non-RAG Response")
                st.write_stream(stream_llm_response(llm, st.session_state.messages))