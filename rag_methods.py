import os
import time
import pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import format_document
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import tempfile
import time
import json
import shutil

load_dotenv()

# Initialize embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"trust_remote_code": True})
embedding_dimension = 384 # Dimension for all-MiniLM-L6-v2

# Initialize Pinecone client
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

pc = Pinecone(api_key=st.secrets.get("PINECONE_API_KEY"))
cloud = st.secrets.get('PINECONE_CLOUD') or 'aws'
region = st.secrets.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

DATA_FOLDER = "rag_data"
METADATA_FILE = os.path.join(DATA_FOLDER, "rag_metadata.json")

# Create the data folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

def initialize_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name):
    """Initializes Pinecone and returns the index."""
    try:
        
        if pinecone_index_name not in pc.list_indexes().names():
            # Create a new index with the correct dimension
            st.write(f"Pinecone index '{pinecone_index_name}' does not exist. Creating it with dimension {embedding_dimension}...")
            pc.create_index(
                index_name=pinecone_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=pinecone_environment
                )
            )
            st.write(f"Pinecone index '{pinecone_index_name}' creation initiated.")
            
            # Wait for the index to be ready
             # Wait for index to be ready
            while not pc.describe_index(pinecone_index_name).status['ready']:
              time.sleep(1)
        
        index = pc.Index(pinecone_index_name)
        vector_db = LangchainPinecone(index=index, embedding=embedding_model, text_key="text") # Create LangchainPinecone object with text_key
        
        # Load persisted documents if they exist
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
            for doc_name, vector_ids in metadata.items():
                st.write(f"Loading persisted document: {doc_name}")
                st.session_state.rag_sources.extend([doc_name])
        
        return vector_db
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None

def load_doc_to_db(pinecone_index, rag_docs, pinecone_index_name):
    """Loads documents into the Pinecone vector database."""
    if not rag_docs:
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    if not os.path.exists(METADATA_FILE):
        metadata = {}
    else:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    
    for doc in rag_docs:
        file_extension = doc.name.split(".")[-1].lower()
        
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp_file:
                tmp_file.write(doc.read())
                tmp_file_path = tmp_file.name
            
            if file_extension == "pdf":
                loader = PyPDFLoader(file_path=tmp_file_path)
            elif file_extension == "txt":
                loader = TextLoader(file_path=tmp_file_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(file_path=tmp_file_path)
            elif file_extension == "md":
                loader = UnstructuredMarkdownLoader(file_path=tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue
            
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            
            vector_ids = pinecone_index.add_documents(documents=chunks) # Use the LangchainPinecone object to add documents
            
            st.session_state.rag_sources.extend([doc.name])
            metadata[doc.name] = vector_ids
            st.success(f"Document '{doc.name}' loaded to DB")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def load_url_to_db(pinecone_index, rag_url, pinecone_index_name):
    """Loads content from a URL into the Pinecone vector database."""
    if not rag_url:
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    if not os.path.exists(METADATA_FILE):
        metadata = {}
    else:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    
    try:
        from langchain.document_loaders import WebBaseLoader
        loader = WebBaseLoader(rag_url)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        
        vector_ids = pinecone_index.add_documents(documents=chunks) # Use the LangchainPinecone object to add documents
        
        st.session_state.rag_sources.extend([rag_url])
        metadata[rag_url] = vector_ids
        st.success(f"URL '{rag_url}' loaded to DB")
    except Exception as e:
        st.error(f"Error loading URL: {e}")
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def stream_llm_response(llm, messages):
    """Streams the LLM response."""
    for chunk in llm.stream(messages):
        yield chunk.content

def stream_llm_rag_response(llm, messages):
    """Streams the LLM response with RAG."""
    
    if not st.session_state.vector_db:
        yield "No vector database initialized."
        return
    
    retriever = st.session_state.vector_db.as_retriever()
    
    template = """
    You are a helpful assistant that answers questions based on the context provided.
    Use the following context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        prompt_template = PromptTemplate.from_template("{page_content}") # Create a simple prompt template
        return "\n\n".join(format_document(doc, prompt=prompt_template) for doc in docs) # Pass the prompt
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    question = messages[-1].content
    
    # Stream the response and yield each chunk
    for chunk in chain.stream(question):
        yield chunk