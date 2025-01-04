import os
import tempfile
import json
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

# --- Constants ---
DATA_FOLDER = "rag_data"  # Folder to store metadata
METADATA_FILE = os.path.join(DATA_FOLDER, "rag_metadata.json")  # File to store document metadata
os.makedirs(DATA_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# --- Initialize Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
embedding_dimension = 384  # Dimension for the embedding model

# --- Document Loaders ---
LOADERS = {
    "pdf": PyPDFLoader,
    "txt": TextLoader,
    "docx": Docx2txtLoader,
    "md": UnstructuredMarkdownLoader,
}

# --- Helper Functions ---
def load_metadata():
    """Loads metadata from the metadata file."""
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  # Return an empty dictionary if the file doesn't exist or is invalid

def save_metadata(metadata):
    """Saves metadata to the metadata file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

# --- Pinecone Initialization ---
def initialize_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name):
    """Initializes Pinecone and returns the vector database."""
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Create the index if it doesn't exist
        if pinecone_index_name not in pc.list_indexes().names():
            st.info(f"Pinecone index '{pinecone_index_name}' does not exist. Creating it with dimension {embedding_dimension}...")
            pc.create_index(
                index_name=pinecone_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=pinecone_environment
                )
            )
            st.success(f"Pinecone index '{pinecone_index_name}' creation initiated.")

            # Wait for the index to be ready
            while not pc.describe_index(pinecone_index_name).status['ready']:
                time.sleep(1)

        # Initialize the Pinecone index
        index = pc.Index(pinecone_index_name)
        vector_db = LangchainPinecone(index=index, embedding=embedding_model, text_key="text")

        # Load persisted documents if they exist
        metadata = load_metadata()
        for doc_name in metadata.keys():
            st.session_state.rag_sources.append(doc_name)  # Add document names to session state
            st.info(f"Loaded persisted document: {doc_name}")

        return vector_db
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None

# --- Document and URL Loading ---
def load_doc_to_db(pinecone_index, rag_docs, pinecone_index_name):
    """Loads documents into the Pinecone vector database."""
    if not rag_docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metadata = load_metadata()  # Load existing metadata

    for doc in rag_docs:
        file_extension = doc.name.split(".")[-1].lower()
        loader_class = LOADERS.get(file_extension)

        if not loader_class:
            st.warning(f"Unsupported file type: {file_extension}")
            continue

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp_file:
            tmp_file.write(doc.read())
            tmp_file_path = tmp_file.name

        try:
            # Load and process the document
            loader = loader_class(file_path=tmp_file_path)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            vector_ids = pinecone_index.add_documents(documents=chunks)

            # Update metadata and session state
            st.session_state.rag_sources.append(doc.name)
            metadata[doc.name] = vector_ids
            st.success(f"Document '{doc.name}' loaded to DB")
        except Exception as e:
            st.error(f"Error loading document '{doc.name}': {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    save_metadata(metadata)  # Save updated metadata

def load_url_to_db(pinecone_index, rag_url, pinecone_index_name):
    """Loads content from a URL into the Pinecone vector database."""
    if not rag_url:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metadata = load_metadata()  # Load existing metadata

    try:
        # Load and process the URL content
        loader = WebBaseLoader(rag_url)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vector_ids = pinecone_index.add_documents(documents=chunks)

        # Update metadata and session state
        st.session_state.rag_sources.append(rag_url)
        metadata[rag_url] = vector_ids
        st.success(f"URL '{rag_url}' loaded to DB")
    except Exception as e:
        st.error(f"Error loading URL: {e}")

    save_metadata(metadata)  # Save updated metadata

# --- LLM Response Streaming ---
def stream_llm_response(llm, messages):
    """Streams the LLM response without RAG."""
    try:
        for chunk in llm.stream(messages):
            # Handle different chunk types
            if isinstance(chunk, str):
                yield chunk  # Yield the string directly
            elif hasattr(chunk, 'content'):
                # Handle cases where the chunk has a 'content' attribute
                if isinstance(chunk.content, list):
                    # If the content is a list, join it into a single string with a separator
                    separator = "\n- "  # Dash separator
                    list_content = separator.join(str(item) for item in chunk.content)
                    yield list_content
                else:
                    yield chunk.content  # Yield the content attribute
            elif isinstance(chunk, dict) and 'content' in chunk:
                # Handle cases where the chunk is a dictionary with a 'content' key
                if isinstance(chunk['content'], list):
                    # If the content is a list, join it into a single string with a separator
                    separator = "\n- "  # Dash separator
                    list_content = separator.join(str(item) for item in chunk['content'])
                    yield list_content
                else:
                    yield chunk['content']  # Yield the content from a dictionary
            elif isinstance(chunk, list):
                # If the chunk is a list, convert it to a string with a separator
                separator = "\n- "  # Dash separator
                list_content = separator.join(str(item) for item in chunk)
                yield list_content
            else:
                # Convert other types to string
                yield str(chunk)
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def stream_llm_rag_response(llm, messages):
    if not st.session_state.vector_db:
        yield "No vector database initialized."
        return
    
    retriever = st.session_state.vector_db.as_retriever()
    prompt = PromptTemplate.from_template("""
        You are a helpful assistant that explains your reasoning before providing a final answer.
        If the question requires context, use the following context to answer the question:
        
        Context:
        {context}
        
        Question: {question}
        
        Reasoning and Answer:
    """)
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    for chunk in chain.stream(messages[-1].content):
        yield chunk.content if hasattr(chunk, 'content') else str(chunk)