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

DB_DOCS_LIMIT = 10
INDEX_NAME = "langchain-rag"

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

def initialize_vector_db(docs: List[Document]) -> LangchainPinecone:
    """Initialize vector database with provided documents"""
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Get both the index and index_name
        pinecone_index, index_name = initialize_pinecone()
        
        return LangchainPinecone.from_documents(
            documents=docs,
            embedding=embedding_function,
            index_name=index_name,
            namespace=f"ns_{st.session_state.session_id}"
        )
        
    except Exception as e:
        st.error(f"Vector DB initialization failed: {str(e)}")
        return None

def process_documents(docs: List[Document]) -> None:
    """Process and load documents into vector database"""
    if not docs:
        return
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            st.warning("No content extracted from documents.")
            return
            
        if st.session_state.vector_db is None:
            vector_db = initialize_vector_db(chunks)
            if vector_db:
                st.session_state.vector_db = vector_db
        else:
            try:
                st.session_state.vector_db.add_documents(chunks)
            except Exception:
                vector_db = initialize_vector_db(chunks)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")

def load_doc_to_db():
    """Load documents to vector database"""
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            # Check if the document already exists in the session or on disk
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                    st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
                    break

                try:
                    # Save the document to a directory on disk
                    doc_dir = Path("docs")
                    doc_dir.mkdir(parents=True, exist_ok=True)
                    doc_path = doc_dir / doc_file.name
                    
                    with open(doc_path, "wb") as f:
                        f.write(doc_file.getvalue())

                    # Load document using appropriate loader
                    loader = None
                    if doc_file.type == "application/pdf":
                        loader = PyPDFLoader(str(doc_path))
                    elif doc_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(str(doc_path))
                    elif doc_file.type in ["text/plain", "text/markdown"]:
                        loader = TextLoader(str(doc_path))

                    if loader:
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                            
                except Exception as e:
                    st.error(f"Error loading {doc_file.name}: {str(e)}")

        if docs:
            process_documents(docs)
            st.success(f"Documents loaded successfully!")

def load_url_to_db():
    """Load URL content to vector database"""
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
                return

            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                st.session_state.rag_sources.append(url)
                process_documents(docs)
                st.success(f"URL content loaded successfully!")
            except Exception as e:
                st.error(f"Error loading URL: {str(e)}")

def initialize_documents():
    """Load documents from disk if they exist"""
    doc_dir = Path("docs")
    if doc_dir.exists():
        for doc_file in doc_dir.iterdir():
            # Initialize rag_sources in session state before loading
            if doc_file.name not in st.session_state.rag_sources:
                try:
                    loader = None
                    if doc_file.suffix == ".pdf":
                        loader = PyPDFLoader(str(doc_file))
                    elif doc_file.suffix == ".docx":
                        loader = Docx2txtLoader(str(doc_file))
                    elif doc_file.suffix in [".txt", ".md"]:
                        loader = TextLoader(str(doc_file))

                    if loader:
                        docs = loader.load()
                        process_documents(docs)
                        st.session_state.rag_sources.append(doc_file.name)

                except Exception as e:
                    st.error(f"Error loading {doc_file.name}: {str(e)}")

def get_rag_chain(llm):
    """Create RAG chain for conversational retrieval"""
    retriever = st.session_state.vector_db.as_retriever(
        search_kwargs={"k": 3}
    )
    
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Generate a search query based on our conversation, focusing on recent messages.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, context_prompt)
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context and your knowledge. Context: {context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}")
    ])
    
    return create_retrieval_chain(
        retriever_chain,
        create_stuff_documents_chain(llm, response_prompt)
    )

def stream_llm_response(llm_stream, messages):
    """Stream LLM response without RAG"""
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_llm_rag_response(llm_stream, messages):
    """Stream RAG-enhanced LLM response"""
    rag_chain = get_rag_chain(llm_stream)
    response_message = "🔍 "
    
    for chunk in rag_chain.pick("answer").stream({
        "messages": messages[:-1],
        "input": messages[-1].content
    }):
        response_message += chunk
        yield chunk
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_message
    })