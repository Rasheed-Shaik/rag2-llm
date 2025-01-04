import streamlit as st
import os
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    initialize_pinecone,
)

# --- Constants ---
MODELS = ["google/gemini-2.0-flash-exp", "think/gemini-2.0-flash-thinking-exp"]

# --- Page Config ---
st.set_page_config(
    page_title="RAG LLM App", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown("""<h2 style="text-align: center;">📚🔍 <i> Do your LLM even RAG bro? </i> 🤖💬</h2>""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I assist you today?"}]

# --- Sidebar ---
with st.sidebar:
    # API Key Inputs
    google_api_key = st.text_input("🔐 Google API Key", type="password", key="google_api_key")
    anthropic_api_key = st.text_input("🔐 Anthropic API Key", type="password", key="anthropic_api_key")
    pinecone_index_name = st.text_input("🗂️ Pinecone Index Name", key="pinecone_index_name")

    # Pinecone Initialization
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if st.session_state.vector_db is None:
        with st.spinner("Initializing Pinecone..."):
            pinecone_init_result = initialize_pinecone(
                st.secrets.get("PINECONE_API_KEY"),
                st.secrets.get("PINECONE_ENVIRONMENT"),
                pinecone_index_name,
            )
            if pinecone_init_result:
                st.session_state.vector_db = pinecone_init_result
                st.success("Pinecone initialized successfully!")
            else:
                st.error("Pinecone initialization failed. Check debug messages.")

    # RAG Toggle and Clear Chat
    is_vector_db_loaded = st.session_state.vector_db is not None
    use_rag = st.toggle("Use RAG", value=is_vector_db_loaded, disabled=not is_vector_db_loaded, key="use_rag")
    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    # RAG Sources
    st.header("RAG Sources:")
    uploaded_files = st.file_uploader("📄 Upload a document", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True, key="rag_docs")
    if uploaded_files:
        load_doc_to_db(st.session_state.vector_db, uploaded_files, pinecone_index_name)

    rag_url = st.text_input("🌐 Introduce a URL", placeholder="https://example.com", key="rag_url")
    if rag_url:
        load_url_to_db(st.session_state.vector_db, rag_url, pinecone_index_name)

    with st.expander(f"📚 Documents in DB ({len(st.session_state.rag_sources)})"):
        st.write(st.session_state.rag_sources)

# --- Main Content ---
if not google_api_key or not pinecone_index_name:
    st.warning("⬅️ Please provide a Google API Key and Pinecone Index Name to continue.")
else:
    # Model Selection
    model_provider = st.selectbox("🤖 Select a Model", options=MODELS).split("/")[0]

    # Initialize LLM
    if model_provider == "google":
        llm_stream = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0,
            streaming=True,
        )
    elif model_provider == "think":
        llm_stream = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp",
            google_api_key=google_api_key,
            temperature=0,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            model="claude-3-opus",
            api_key=anthropic_api_key,
            temperature=0.3,
            streaming=True,
        )

    # Display Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
if prompt := st.chat_input("Your message"):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Convert session messages to HumanMessage/AIMessage format
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" 
            else AIMessage(content=m["content"]) 
            for m in st.session_state.messages
        ]

        try:
            if not st.session_state.use_rag:
                # Stream response without RAG
                for chunk in stream_llm_response(llm_stream, messages):
                    if isinstance(chunk, str):  # Ensure the chunk is a string
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
            else:
                # Stream response with RAG
                for chunk in stream_llm_rag_response(llm_stream, messages):
                    if isinstance(chunk, str):  # Ensure the chunk is a string
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            # Handle errors during streaming
            full_response = f"An error occurred: {str(e)}"
            message_placeholder.markdown(full_response)

        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Display final response
        message_placeholder.markdown(full_response)