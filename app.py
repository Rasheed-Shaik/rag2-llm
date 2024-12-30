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
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
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
        {"role": "assistant", "content": "Hi there! I'm your AI Knowledge Assistant. How can I help you today?"}
    ]

# Page header with enhanced styling
st.markdown("""
    <h2 style="text-align: center; margin-bottom: 1rem;">
        🤖 AI Knowledge Assistant
    </h2>
    <p style="text-align: center; color: #666; margin-bottom: 2rem;">
        Powered by RAG Technology 📚✨
    </p>
""", unsafe_allow_html=True)

# Sidebar configuration with enhanced icons
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key Management with enhanced security icon
    google_api_key = st.secrets.get("google_api_key", "") if hasattr(st, "secrets") else ""
    
    if not google_api_key:
        with st.expander("🔐 API Settings"):
            google_api_key = st.text_input(
                "Google API Key",
                type="password",
                key="google_api_key",
                help="Enter your Google API key here"
            )

    # Enhanced Controls Section
    st.markdown("### 🎮 Controls")
    
    # Model Selection and RAG Toggle
    model = "google/gemini-2.0-flash-thinking-exp"
    st.session_state.use_rag = st.toggle(
        "🧠 Enable Knowledge Base",
        value=st.session_state.vector_db is not None,
        disabled=st.session_state.vector_db is None,
        help="Toggle to enable or disable RAG capabilities"
    )
    
    if st.button("🧹 Clear Chat", type="primary"):
        st.session_state.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! I'm your AI Knowledge Assistant. How can I help you today?"}
        ]
        st.rerun()

    # Knowledge Base Section with enhanced icons
    st.markdown("### 📚 Knowledge Base")
    
    # Document Upload with multiple file types
    st.markdown("""
        <style>
            .upload-text { font-size: 0.9rem; color: #666; }
        </style>
    """, unsafe_allow_html=True)
    
    st.file_uploader(
        "📄 Upload Documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
        help="Upload PDF, TXT, DOCX, or MD files"
    )

    # URL Input with enhanced styling
    st.text_input(
        "🌐 Add Website URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
        help="Enter a website URL to add to the knowledge base"
    )

    # Source Display with enhanced visualization
    with st.expander(f"📂 Knowledge Sources ({len(st.session_state.rag_sources)})"):
        if st.session_state.rag_sources:
            for source in st.session_state.rag_sources:
                if source.startswith('http'):
                    st.markdown(f"🌐 `{source}`")
                else:
                    st.markdown(f"📄 `{source}`")
        else:
            st.markdown("_No sources added yet_")

# Main chat interface
if not google_api_key:
    st.warning("🔑 No Google API Key found. Please add it to your Streamlit secrets or enter it in the sidebar.")
else:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=model.split("/")[-1],
        google_api_key=google_api_key,
        temperature=0,
        streaming=True
    )

    # Display chat messages with enhanced styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    # Chat input and response
    if prompt := st.chat_input("💭 Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages
            ]
            
            if st.session_state.use_rag:
                st.write_stream(stream_llm_rag_response(llm, messages))
            else:
                st.write_stream(stream_llm_response(llm, messages))

    # Add footer
    st.markdown("""
        <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background: rgba(255,255,255,0.9);'>
            <p style='color: #666; font-size: 0.8rem;'>
                📚 Powered by RAG Technology | 🤖 Using Google's Gemini Pro
            </p>
        </div>
    """, unsafe_allow_html=True)