import streamlit as st
import os
import tempfile
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone, ServerlessSpec
import json

DB_DOCS_LIMIT = 10
INDEX_NAME = "langchain-rag"
METADATA_NAMESPACE = "document_metadata"

def initialize_pinecone():
    """Initialize Pinecone client using the new Pinecone class"""
    pc = Pinecone(
        api_key=st.secrets.get("PINECONE_API_KEY")
    )
    
    # Create index if it doesn't exist
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # dimension for BAAI text embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Get the index
    index = pc.Index(INDEX_NAME)
    return index, INDEX_NAME

def get_embedding_function():
    """Get the embedding function"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

def save_document_metadata(doc_name: str, doc_type: str):
    """Save document metadata to Pinecone"""
    try:
        embedding_function = get_embedding_function()
        
        metadata = {
            "name": doc_name,
            "type": doc_type,
            "session_id": st.session_state.session_id
        }
        
        # Create a metadata document
        metadata_doc = Document(
            page_content=json.dumps(metadata),
            metadata={"namespace": METADATA_NAMESPACE}
        )
        
        # Initialize vector store for metadata if not exists
        if "metadata_store" not in st.session_state:
            _, index_name = initialize_pinecone()
            st.session_state.metadata_store = LangchainPinecone.from_documents(
                documents=[metadata_doc],
                embedding=embedding_function,
                index_name=index_name,
                namespace=METADATA_NAMESPACE
            )
        else:
            st.session_state.metadata_store.add_documents([metadata_doc])
            
    except Exception as e:
        st.error(f"Error saving document metadata: {str(e)}")

def load_persisted_documents():
    """Load document metadata from Pinecone"""
    try:
        if "metadata_store" not in st.session_state:
            embedding_function = get_embedding_function()
            index, _ = initialize_pinecone()
            
            st.session_state.metadata_store = LangchainPinecone(
                embedding=embedding_function,
                pinecone_index=index,
                namespace=METADATA_NAMESPACE
            )
        
        # Query all documents for the current session
        results = st.session_state.metadata_store.similarity_search(
            "document metadata",
            filter={"session_id": st.session_state.session_id}
        )
        
        # Extract document names from metadata
        for result in results:
            metadata = json.loads(result.page_content)
            if metadata["name"] not in st.session_state.rag_sources:
                st.session_state.rag_sources.append(metadata["name"])
                
    except Exception as e:
        st.error(f"Error loading persisted documents: {str(e)}")

def initialize_vector_db(docs: List[Document]) -> LangchainPinecone:
    """Initialize vector database with provided documents"""
    try:
        embedding_function = get_embedding_function()
        index, index_name = initialize_pinecone()
        
        return LangchainPinecone.from_documents(
            documents=docs,
            embedding=embedding_function,
            index_name=index_name,
            namespace=f"ns_{st.session_state.session_id}"
        )
        
    except Exception as e:
        st.error(f"Vector DB initialization failed: {str(e)}")
        return None

# Rest of the code remains the same...