import streamlit as st
import os
import uuid
import platform

# SQLite fix for Streamlit Cloud
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
from rag_methods import stream_llm_response, stream_llm_rag_response, load_doc_to_db, load_url_to_db, initialize_llm

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Chat App",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session states
if "session_id" not in st.session_state:
    st.session_state.update({
        "session_id": str(uuid.uuid4()),
        "rag_sources": [],
        "vector_db": None,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ],
        "use_rag": False,
        "model": "google/gemini-2.0-flash-exp"
    })

# Custom CSS for chat messages
st.markdown("""
    <style>

    /* Dark mode adjustments */
    [data-theme="dark"] .chat-message.user-message {
        background-color: #262a34; /* Darker background for user messages */
        border: 1px solid #4a5568; /* Darker border for user messages */
        color: #f0f0f0; /* Light text for contrast */
    }

    [data-theme="dark"] .chat-message.assistant-message {
        background-color: #374151; /* Darker background for assistant messages */
        border: 1px solid #4a5568; /* Darker border for assistant messages */
        color: #f0f0f0; /* Light text for contrast */
    }
    [data-theme="dark"] .stTextInput > div > div > input,
    [data-theme="dark"] .stTextArea > div > div > textarea {
        background-color: #374151 !important;
        color: #f0f0f0 !important;
        border-color: #4a5568 !important;
    }
    [data-theme="dark"] .stSelectbox > div > div > div,
    [data-theme="dark"] .stFileUploader > div > div,
    [data-theme="dark"] .stExpander > div > div > div{
        background-color: #374151 !important;
        color: #f0f0f0 !important;
    }
    [data-theme="dark"] .stButton > button {
        background-color: #4a5568 !important;
        color: #f0f0f0 !important;
    }

    .chat-message {
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e6f7ff;
        border: 1px solid #91caff;
        text-align: right;
    }
    .assistant-message {
        background-color: #f0f0f0;
        border: 1px solid #d3d3d3;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""<h2 style="text-align: center;">🌍✨ RAG-Enabled Chat Assistant 🧠🦾 </h2>""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    google_api_key = st.secrets.get("google_api_key", "")

    # Only show API input if no key in secrets
    if not google_api_key:
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            key="google_api_key"
        )
    st.markdown("### 🎮 Controls")
    # Model Selection
    model = st.selectbox(
        "Select Model",
        ["google/gemini-2.0-flash-exp", "google/gemini-pro"],
        index = 0 if st.session_state.model == "google/gemini-2.0-flash-exp" else 1,
        key="model_selection"
    )
    st.session_state.model = model


    # Initialize RAG toggle based on vector_db presence
    st.session_state.use_rag = st.session_state.vector_db is not None
    st.session_state.use_rag = st.toggle(
        "Enable RAG",
        value=st.session_state.use_rag,
        disabled=st.session_state.vector_db is None
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! I'm your AI Knowledge Assistant. How can I help you today?"}
        ]
        st.rerun()
    # RAG Document Management
    st.header("🏫 Knowledge Base")
    st.file_uploader(
        "Upload Documents 📚",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
        help="Upload documents to enhance the knowledge base."
    )

    st.text_input(
        "Add Website URL 🔗",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
        help="Add a URL to fetch content from a website."
    )

    with st.expander(f"📂 Loaded Sources ({len(st.session_state.rag_sources)})"):
         if st.session_state.rag_sources:
            for source in st.session_state.rag_sources:
                st.markdown(f"- {source}")
         else:
                st.markdown("No sources loaded yet.")

# Main chat interface
if not google_api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to continue.")
else:
    # Initialize LLM
    llm = initialize_llm(st.session_state.model, google_api_key)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # Chat input and response
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user-message">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
           if st.session_state.use_rag:
                if st.session_state.vector_db:
                     with st.spinner("Thinking..."):
                         messages = [
                            HumanMessage(content=m["content"]) if m["role"] == "user"
                            else AIMessage(content=m["content"])
                            for m in st.session_state.messages
                        ]
                         response_placeholder = st.empty()
                         full_response = ""
                         for chunk in stream_llm_rag_response(llm, messages):
                            full_response += chunk
                            response_placeholder.markdown(f'<div class="chat-message assistant-message">{full_response}</div>', unsafe_allow_html=True)

                else:
                     st.warning("Please upload documents or URLs to use RAG.")
           else:
                with st.spinner("Thinking..."):
                    messages = [
                        HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"])
                        for m in st.session_state.messages
                    ]
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream_llm_response(llm, messages):
                        full_response += chunk
                        response_placeholder.markdown(f'<div class="chat-message assistant-message">{full_response}</div>', unsafe_allow_html=True)