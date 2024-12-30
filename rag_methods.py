# rag_methods.py
import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from pathlib import Path

# Langchain imports
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.config import Settings

DB_DOCS_LIMIT = 10
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "chroma_db" # Define persistent directory

def initialize_llm(model: str, google_api_key: str) -> ChatGoogleGenerativeAI:
    """Initialize the LLM."""
    return ChatGoogleGenerativeAI(
        model=model.split("/")[-1],
        google_api_key=google_api_key,
        temperature=0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        streaming=True
    )

def stream_llm_response(llm_stream: ChatGoogleGenerativeAI, messages: List[Any]):
    """Stream LLM response without RAG."""
    full_response = ""
    for chunk in llm_stream.stream(messages):
        text_content = ""
        if hasattr(chunk, "content"):
            if isinstance(chunk.content, list):
                text_content = "".join(str(item) for item in chunk.content)
            else:
                text_content = str(chunk.content)
        elif isinstance(chunk, dict) and "text" in chunk:
            text_content = chunk["text"]
        elif isinstance(chunk, dict) and "candidates" in chunk and chunk["candidates"]:
            best_candidate = chunk["candidates"][0]
            if "content" in best_candidate and "parts" in best_candidate["content"]:
                text_content = "".join(part["text"] for part in best_candidate["content"]["parts"])

        if text_content:
            full_response += text_content
            yield text_content

    st.session_state.messages.append({"role": "assistant", "content": full_response})

def initialize_vector_db(persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma | None:
    """Initialize or load the vector database."""
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )
    chroma_settings = Settings(
        is_persistent=True,
        persist_directory=persist_directory,
        anonymized_telemetry=False,
    )
    try:
        # Try loading the existing database
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            client_settings=chroma_settings,
            collection_name=f"collection_{st.session_state.session_id}",
        )
        return vector_db
    except ValueError:
        # If the database doesn't exist, return None
        return None

def load_rag_sources(vector_db: Chroma) -> List[str]:
    """Load the list of RAG sources from the vector database."""
    if vector_db is not None:
        metadatas = vector_db.get(include=['metadatas'])['metadatas']
        sources = set()
        for metadata in metadatas:
            if 'source' in metadata:
                sources.add(metadata['source'])
        return list(sources)
    return []

def process_documents(docs: List[Document]) -> None:
    """Process and load documents into vector database."""
    if not docs:
        st.warning("No documents to process.")
        return

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            st.warning("No content extracted from documents.")
            return

        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
        chroma_settings = Settings(
            is_persistent=True,
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
        )

        if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
            # Initialize a new database if it doesn't exist
            st.session_state.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                collection_name=f"collection_{st.session_state.session_id}",
                persist_directory=CHROMA_PERSIST_DIR,
                client_settings=chroma_settings,
                metadatas=[{"source": doc.metadata.get("source", "unknown")} for doc in docs] # Store source in metadata
            )
        else:
            try:
                # Load the existing database and add documents
                st.session_state.vector_db = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=embedding_function,
                    client_settings=chroma_settings,
                    collection_name=f"collection_{st.session_state.session_id}",
                )
                st.session_state.vector_db.add_documents(
                    chunks,
                    metadatas=[{"source": doc.metadata.get("source", "unknown")} for doc in docs] # Store source in metadata
                )
            except Exception as e:
                st.error(f"Error adding documents to existing vector DB: {e}")
                # Fallback to creating a new one if loading fails
                st.session_state.vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_function,
                    collection_name=f"collection_{st.session_state.session_id}",
                    persist_directory=CHROMA_PERSIST_DIR,
                    client_settings=chroma_settings,
                    metadatas=[{"source": doc.metadata.get("source", "unknown")} for doc in docs] # Store source in metadata
                )

        st.session_state.vector_db.persist()

    except Exception as e:
        st.error(f"Document processing error: {e}")

def load_doc_to_db(uploaded_files):
    """Load documents to vector database."""
    for uploaded_file in uploaded_files:
        source_name = uploaded_file.name
        if source_name not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
                break

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source_name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name

                    loaders: Dict[str, Any] = {
                        ".pdf": PyPDFLoader,
                        ".docx": Docx2txtLoader,
                        ".txt": TextLoader,
                        ".md": TextLoader,
                    }
                    file_extension = Path(source_name).suffix.lower()
                    loader_class = loaders.get(file_extension)

                    if loader_class:
                        loader = loader_class(file_path)
                        loaded_docs = loader.load()
                        for doc in loaded_docs:
                            doc.metadata['source'] = source_name # Add source to metadata
                        process_documents(loaded_docs)
                        st.session_state.rag_sources.append(source_name)
                    else:
                        st.warning(f"Unsupported file type: {source_name}")

            except Exception as e:
                st.error(f"Error loading {source_name}: {e}")
            finally:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    if uploaded_files:
        st.success("Documents loaded successfully!")

def load_url_to_db(url):
    """Load URL content to vector database."""
    if url and url not in st.session_state.rag_sources:
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
            return
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = url # Add source to metadata
            process_documents(docs)
            st.session_state.rag_sources.append(url)
            st.success("URL content loaded successfully!")
        except Exception as e:
            st.error(f"Error loading URL: {e}")

def get_rag_chain(llm: ChatGoogleGenerativeAI):
    """Create RAG chain for conversational retrieval."""
    if not st.session_state.vector_db_ready or st.session_state.vector_db is None:
        st.error("Vector database is not initialized.")
        return None

    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

    context_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant, use the context to answer the user's question."
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Generate a search query based on our conversation, focusing on recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, context_prompt)

    response_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Answer based on the context and your knowledge. Context: {context}"
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, response_prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm: ChatGoogleGenerativeAI, messages: List[Any]):
    """Stream RAG-enhanced LLM response."""
    rag_chain = get_rag_chain(llm)
    if rag_chain is None:
        return

    for chunk in rag_chain.stream({"messages": messages[:-1], "input": messages[-1].content}):
        if isinstance(chunk, dict) and "answer" in chunk:
            text_content = chunk["answer"]
            yield text_content
        elif isinstance(chunk, str):
            yield chunk