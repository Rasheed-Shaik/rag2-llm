import os
import tempfile
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from pinecone import Pinecone
import streamlit as st

# --- Constants ---
DATA_FOLDER = "rag_data"
METADATA_FILE = os.path.join(DATA_FOLDER, "rag_metadata.json")
os.makedirs(DATA_FOLDER, exist_ok=True)

# --- Initialize Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})

# --- Pinecone Initialization ---
def initialize_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name):
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if pinecone_index_name not in pc.list_indexes().names():
            pc.create_index(
                index_name=pinecone_index_name,
                dimension=384,
                metric="cosine",
                spec={"cloud": "aws", "region": pinecone_environment}
            )
            while not pc.describe_index(pinecone_index_name).status['ready']:
                time.sleep(1)
        
        index = pc.Index(pinecone_index_name)
        vector_db = LangchainPinecone(index=index, embedding=embedding_model, text_key="text")
        
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
            st.session_state.rag_sources.extend(metadata.keys())
        
        return vector_db
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None

# --- Document and URL Loading ---
def load_doc_to_db(pinecone_index, rag_docs, pinecone_index_name):
    if not rag_docs:
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metadata = json.load(open(METADATA_FILE)) if os.path.exists(METADATA_FILE) else {}

    for doc in rag_docs:
        file_extension = doc.name.split(".")[-1].lower()
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp_file:
                tmp_file.write(doc.read())
                tmp_file_path = tmp_file.name

            loader = {
                "pdf": PyPDFLoader,
                "txt": TextLoader,
                "docx": Docx2txtLoader,
                "md": UnstructuredMarkdownLoader,
            }.get(file_extension)

            if not loader:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            documents = loader(tmp_file_path).load()
            chunks = text_splitter.split_documents(documents)
            vector_ids = pinecone_index.add_documents(chunks)
            
            st.session_state.rag_sources.append(doc.name)
            metadata[doc.name] = vector_ids
            st.success(f"Document '{doc.name}' loaded to DB")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def load_url_to_db(pinecone_index, rag_url, pinecone_index_name):
    if not rag_url:
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metadata = json.load(open(METADATA_FILE)) if os.path.exists(METADATA_FILE) else {}

    try:
        documents = WebBaseLoader(rag_url).load()
        chunks = text_splitter.split_documents(documents)
        vector_ids = pinecone_index.add_documents(chunks)
        
        st.session_state.rag_sources.append(rag_url)
        metadata[rag_url] = vector_ids
        st.success(f"URL '{rag_url}' loaded to DB")
    except Exception as e:
        st.error(f"Error loading URL: {e}")
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

# --- LLM Response Streaming ---
def stream_llm_response(llm, messages):
    try:
        for chunk in llm.stream(messages):
            yield chunk.content if hasattr(chunk, 'content') else str(chunk)
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