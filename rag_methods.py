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
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chromadb.config import Settings

DB_DOCS_LIMIT = 10

def stream_llm_response(llm_stream, messages):
    """Stream LLM response without RAG"""
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def initialize_vector_db(docs: List[Document]) -> Chroma:
    """Initialize vector database with provided documents"""
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-base-en",  # Using base model for efficiency
            model_kwargs={"trust_remote_code": True}
        )
        
        temp_dir = tempfile.mkdtemp()
        chroma_settings = Settings(
            is_persistent=True,
            persist_directory=temp_dir,
            anonymized_telemetry=False
        )
        
        return Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            collection_name=f"collection_{st.session_state.session_id}",
            persist_directory=temp_dir,
            client_settings=chroma_settings
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
            chunk_size=1000,  # Reduced chunk size for better processing
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
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                    st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
                    break

                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(doc_file.getvalue())
                        file_path = tmp_file.name

                        loader = None
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)

                        if loader:
                            docs.extend(loader.load())
                            st.session_state.rag_sources.append(doc_file.name)
                            
                except Exception as e:
                    st.error(f"Error loading {doc_file.name}: {str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.unlink(file_path)

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