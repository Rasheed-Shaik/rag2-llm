# app.py
import streamlit as st
import os
import uuid
import platform
from pathlib import Path

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

# Local module import
from rag_methods import stream_llm_response, stream_llm_rag_response, load_doc_to_db, load_url_to_db, initialize_vector_db

# Streamlit page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# SQLite fix for Streamlit Cloud
if platform.system() != "Windows":
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

# Initialize session states
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.rag_sources = []
    st.session_state.vector_db_ready = False  # Flag to indicate if vector DB is ready
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! I'm your AI Knowledge Assistant. How can I help you today?"}
    ]

# --- Header Section ---
st.markdown("""
    <div style="text-align: center;">
        <h2>🤖 AI Knowledge Assistant</h2>
        <p style="color: #666;">Powered by RAG Technology 📚✨</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key Management
    google_api_key = st.secrets.get("google_api_key") if "google_api_key" in st.secrets else st.session_state.get("google_api_key", "")

    if not google_api_key:
        with st.expander("🔐 API Settings"):
            google_api_key = st.text_input(
                "Google API Key",
                type="password",
                key="google_api_key",
                help="Enter your Google API key here"
            )
            st.session_state.google_api_key = google_api_key  # Store in session state

    st.markdown("### 🎮 Controls")

    # Model Selection (Consider making this configurable if needed)
    model = "google/gemini-2.0-flash-thinking-exp-1219"
    st.session_state.use_rag = st.toggle(
        "🧠 Enable Knowledge Base",
        value=st.session_state.vector_db_ready,
        disabled=not st.session_state.vector_db_ready,
        help="Toggle to enable or disable RAG capabilities"
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! I'm your AI Knowledge Assistant. How can I help you today?"}
        ]
        st.rerun()

    st.markdown("### 📚 Knowledge Base")

    # Document Upload
    uploaded_files = st.file_uploader(
        "📄 Upload Documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="rag_docs",
        help="Upload PDF, TXT, DOCX, or MD files"
    )

    # Trigger document loading on upload
    if uploaded_files:
        with st.spinner("Loading documents..."):
            load_doc_to_db(uploaded_files)
            if not st.session_state.vector_db_ready and st.session_state.rag_sources:
                st.session_state.vector_db_ready = True
                st.rerun()

    # URL Input
    url_input = st.text_input(
        "🌐 Add Website URL",
        placeholder="https://example.com",
        key="rag_url",
        help="Enter a website URL to add to the knowledge base"
    )

    # Trigger URL loading on input change
    if st.session_state.rag_url != url_input and url_input:
        with st.spinner("Loading URL..."):
            load_url_to_db(url_input)
            if not st.session_state.vector_db_ready and st.session_state.rag_sources:
                st.session_state.vector_db_ready = True
                st.rerun()
        st.session_state.rag_url = url_input # Update session state to avoid re-triggering

    # Source Display
    with st.expander(f"📂 Knowledge Sources ({len(st.session_state.rag_sources)})"):
        if st.session_state.rag_sources:
            for source in st.session_state.rag_sources:
                icon = "🌐" if source.startswith('http') else "📄"
                st.markdown(f"{icon} `{source}`")
        else:
            st.markdown("_No sources added yet_")

# --- Main Chat Interface ---
if not google_api_key:
    st.warning("🔑 No Google API Key found. Please add it to your Streamlit secrets or enter it in the sidebar.")
elif not st.session_state.vector_db_ready and st.session_state.rag_sources:
    with st.spinner("Initializing Knowledge Base... This may take a moment."):
        st.session_state.vector_db = initialize_vector_db(persist_directory="chroma_db")
        st.session_state.vector_db_ready = True
        st.session_state.use_rag = True # Enable RAG once DB is ready
        st.rerun() # Rerun to update UI
else:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=model.split("/")[-1],
        google_api_key=google_api_key,
        temperature=0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        streaming=True
    )

    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input and response
    if prompt := st.chat_input("💭 Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            if st.session_state.use_rag and st.session_state.vector_db_ready:
                response_stream = stream_llm_rag_response(llm, messages)
                if response_stream:
                    for chunk in response_stream:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            else:
                response_stream = stream_llm_response(llm, messages)
                if response_stream:
                    for chunk in response_stream:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

# --- Footer ---
st.markdown("""
    <div style='text-align: center; padding: 10px; margin-top: 2rem; border-top: 1px solid #eee;'>
        <p style='color: #666; font-size: 0.8rem;'>
            📚 Powered by RAG Technology | 🤖 Using Google's Gemini Pro
        </p>
    </div>
""", unsafe_allow_html=True)