import streamlit as st
import os
import dotenv
import uuid

# check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage,AIMessage
from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    initialize_pinecone,
)

dotenv.load_dotenv()


MODELS = ["google/gemini-2.0-flash-exp"]


st.set_page_config(
    page_title="RAG LLM app?", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown("""<h2 style="text-align: center;">📚🔍 <i> Do your LLM even RAG bro? </i> 🤖💬</h2>""", unsafe_allow_html=True)

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]
    
# --- Side Bar LLM API Tokens ---  
with st.sidebar:
        default_google_api_key = os.getenv("google_api_key") if os.getenv("google_api_key") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 Google"):
            google_api_key = st.text_input(
                "Input your Google API Key", 
                value=default_google_api_key, 
                type="password",
                key="google_api_key",
            )

        default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        with st.popover("🔐 Anthropic"):
            anthropic_api_key = st.text_input(
                "Introduce your Anthropic API Key (https://console.anthropic.com/)", 
                value=default_anthropic_api_key, 
                type="password",
                key="anthropic_api_key",
            )
        
        # Pinecone API key and environment are now loaded from st.secrets
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
        pinecone_environment = st.secrets.get("PINECONE_ENVIRONMENT")
        
        default_pinecone_index_name = os.getenv("PINECONE_INDEX_NAME") if os.getenv("PINECONE_INDEX_NAME") is not None else ""
        with st.popover("🗂️ Pinecone Index Name"):
            pinecone_index_name = st.text_input(
                "Introduce your Pinecone Index Name",
                value=default_pinecone_index_name,
                key="pinecone_index_name",
            )
    


# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_google = google_api_key == "" or google_api_key is None
missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None
missing_pinecone = pinecone_api_key is None or pinecone_environment is None or pinecone_index_name == ""
if missing_google or missing_pinecone:
    st.write("#")
    st.warning("⬅️ Please introduce an API Key to continue...")
else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "google" in model and not missing_google:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)
            

        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        if "vector_db" not in st.session_state:
          st.session_state.vector_db = None
        
        # Initialize Pinecone on startup
        if st.session_state.vector_db is None:
            with st.spinner("Initializing Pinecone..."):
                st.write("Attempting to initialize Pinecone...")
                pinecone_init_result = initialize_pinecone(
                    pinecone_api_key,
                    pinecone_environment,
                    pinecone_index_name,
                )
                if pinecone_init_result:
                    st.session_state.vector_db = pinecone_init_result
                    st.success("Pinecone initialized successfully!")
                else:
                    st.error("Pinecone initialization failed. Check debug messages.")
        
        with cols0[0]:
            is_vector_db_loaded = (st.session_state.vector_db is not None)
            st.toggle(
              "Use RAG", 
              value=is_vector_db_loaded, 
              key="use_rag", 
              disabled=not is_vector_db_loaded,
)
        with cols0[1]:
                st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")
        
        st.header("RAG Sources:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # URL input for RAG with websites
        st.text_input(
            "🌐 Introduce a URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    
    
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=google_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream= ChatAnthropic(
            api_key=anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "google":
        llm_stream = ChatGoogleGenerativeAI(
            model=st.session_state.model.split("/")[-1],
            google_api_key=google_api_key,
            temperature=0.3,
            streaming=True,
            
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))