# rag_methods.py
import os
import tempfile
from typing import List
from pathlib import Path
from langchain.schema import Document, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone  # Add this import statement
from pinecone import ServerlessSpec
import json
import streamlit as st  # Keep this for potential caching decorators

DB_DOCS_LIMIT = 10
INDEX_NAME = "langchain-rag"
METADATA_NAMESPACE = "document_metadata"
PERSIST_DIRECTORY = "rag_chroma_db" # Example for local persistence

# SQLite fix - keep this here if needed for functions in this file
import platform
if platform.system() != "Windows":
    try:
        import pysqlite3
        import sys
        if 'pysqlite3' in sys.modules:
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        else:
            print("rag_methods.py: Warning: pysqlite3 was imported but not found in sys.modules.")
    except ImportError:
        print("rag_methods.py: Warning: pysqlite3 not imported. This might cause issues with SQLite on Streamlit Cloud.")
    pass
@st.cache_resource()
def initialize_pinecone():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
    if not pinecone_api_key or not pinecone_environment:
        print("rag_methods.py: Pinecone API key and environment not found in environment variables.")
        return None  # Or handle this case as needed

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    if INDEX_NAME not in pinecone.list_indexes():
        print(f"rag_methods.py: Creating Pinecone index '{INDEX_NAME}'...")
        pinecone.create_index(
            INDEX_NAME,
            dimension=768,  # Adjust based on your embedding model
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust region as needed
        )
    return pinecone.Index(INDEX_NAME)

@st.cache_resource()
def get_embedding_function():
    try:
        return HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    except Exception as e:
        print(f"rag_methods.py: Error initializing embedding function: {e}")
        return None

def save_document_metadata(doc_name: str, doc_type: str):
    # In a real application, you'd persist this metadata (e.g., to a database)
    print(f"rag_methods.py: Saving metadata for {doc_name} of type {doc_type}")
    if "rag_sources" in st.session_state:
        st.session_state.rag_sources.append(f"{doc_name} ({doc_type})")

def get_metadata_store():
    # Logic to retrieve metadata, if needed
    return None

def load_persisted_documents():
    # Logic to load persisted documents, if applicable
    return []

def initialize_documents():
    print("rag_methods.py: initialize_documents() called")
    # This function might be responsible for loading initial documents or setting up the vector DB
    # For now, let's just print a message
    print("rag_methods.py: No initial document loading configured in initialize_documents().")

def initialize_vector_db(docs: List[Document]) -> LangchainPinecone:
    print("rag_methods.py: initialize_vector_db() called")
    pinecone_index = initialize_pinecone()
    embeddings_function = get_embedding_function()

    if pinecone_index is None or embeddings_function is None:
        print("rag_methods.py: Pinecone or embedding function not initialized. Cannot initialize vector DB.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    try:
        vector_db = LangchainPinecone.from_documents(
            chunks, embeddings_function, index_name=INDEX_NAME
        )
        print("rag_methods.py: Vector database initialized successfully.")
        return vector_db
    except Exception as e:
        print(f"rag_methods.py: Error initializing vector database: {e}")
        return None

def process_documents(docs: List[Document], doc_name: str, doc_type: str) -> None:
    print(f"rag_methods.py: Processing document: {doc_name}")
    save_document_metadata(doc_name, doc_type)
    vector_db = initialize_vector_db(docs)
    if vector_db:
        st.session_state.vector_db = vector_db
        print(f"rag_methods.py: Vector database updated in session state.")

def load_doc_to_db(uploaded_files):
    print("rag_methods.py: load_doc_to_db() called")
    embeddings_function = get_embedding_function()
    if not embeddings_function:
        st.error("Embedding function not initialized.")
        return

    all_docs = []
    for uploaded_file in uploaded_files:
        file_extension = Path(uploaded_file.name).suffix.lower()
        file_name = Path(uploaded_file.name).stem
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(tmp_file_path)
            elif file_extension == ".txt" or file_extension == ".md":
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue
            docs = loader.load()
            all_docs.extend(docs)
            save_document_metadata(uploaded_file.name, file_extension[1:]) # Remove the dot
        except Exception as e:
            st.error(f"Error loading document {uploaded_file.name}: {e}")
        finally:
            os.remove(tmp_file_path)

    if all_docs:
        process_documents(all_docs, "uploaded_files", "multiple")
    else:
        print("rag_methods.py: No documents loaded.")

def load_url_to_db(url):
    print("rag_methods.py: load_url_to_db() called")
    embeddings_function = get_embedding_function()
    if not embeddings_function:
        st.error("Embedding function not initialized.")
        return

    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        process_documents(docs, url, "webpage")
    except Exception as e:
        st.error(f"Error loading URL {url}: {e}")

def get_rag_chain(llm):
    # Implement your RAG chain logic here
    # This might involve retrieving relevant documents from the vector DB
    # and then passing them to the LLM
    return None

def stream_llm_response(llm_stream, messages: List[HumanMessage]):
    combined_content = ""
    for chunk in llm_stream.stream(messages):
        combined_content += chunk.content
        yield combined_content

def stream_llm_rag_response(llm_stream, messages: List[HumanMessage]):
    if st.session_state.vector_db is not None:
        last_message_content = messages[-1].content
        print(f"rag_methods.py: Performing RAG query: {last_message_content}")
        retriever = st.session_state.vector_db.as_retriever()
        relevant_docs = retriever.get_relevant_documents(last_message_content)
        print(f"rag_methods.py: Retrieved {len(relevant_docs)} documents.")

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt_with_context = f"Answer the following question based on this context:\n\n{context}\n\nQuestion: {last_message_content}"

        # For simplicity, we're just modifying the last message. A more robust approach
        # might involve a proper RAG chain.
        modified_messages = messages[:-1] + [HumanMessage(content=prompt_with_context)]

        combined_content = ""
        for chunk in llm_stream.stream(modified_messages):
            combined_content += chunk.content
            yield combined_content
    else:
        yield "RAG is enabled, but the knowledge base is empty. Please upload documents or add a URL."