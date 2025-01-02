# rag_methods.py
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
    return pc.Index(INDEX_NAME)

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
        index = initialize_pinecone()

        metadata = {
            "name": doc_name,
            "type": doc_type,
            "session_id": st.session_state.session_id
        }

        metadata_doc = Document(
            page_content=json.dumps(metadata),
            metadata={"source": doc_name} # Add source for filtering
        )

        vectorstore = LangchainPinecone(index, embedding_function, METADATA_NAMESPACE, text_key="page_content")
        vectorstore.add_documents([metadata_doc])

    except Exception as e:
        st.error(f"Error saving document metadata: {str(e)}")

def load_persisted_documents():
    """Load document metadata from Pinecone"""
    try:
        embedding_function = get_embedding_function()
        index = initialize_pinecone()

        st.session_state.metadata_store = LangchainPinecone(
            index,
            embedding=embedding_function,
            namespace=METADATA_NAMESPACE,
            text_key="page_content"  # Add text_key here
        )

        # Query all documents for the current session
        results = st.session_state.metadata_store.similarity_search(
            "document metadata",
            k=100, # Fetch a reasonable number of results
            filter={"session_id": st.session_state.session_id}
        )

        # Extract document names from metadata
        for result in results:
            try:
                metadata = json.loads(result.page_content)
                if metadata["name"] not in st.session_state.rag_sources:
                    st.session_state.rag_sources.append(metadata["name"])
            except json.JSONDecodeError:
                st.error(f"Error decoding metadata: {result.page_content}")

    except Exception as e:
        st.error(f"Error loading persisted documents: {str(e)}")

def initialize_vector_db(docs: List[Document]) -> LangchainPinecone:
    """Initialize vector database with provided documents"""
    try:
        embedding_function = get_embedding_function()
        index = initialize_pinecone()

        return LangchainPinecone.from_documents(
            documents=docs,
            embedding=embedding_function,
            index=index,
            namespace=f"ns_{st.session_state.session_id}",
            text_key="page_content" # Add text_key here
        )

    except Exception as e:
        st.error(f"Vector DB initialization failed: {str(e)}")
        return None

def process_documents(docs: List[Document], doc_name: str, doc_type: str) -> None:
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
                save_document_metadata(doc_name, doc_type)
        else:
            try:
                embedding_function = get_embedding_function()
                index = initialize_pinecone()
                namespace = f"ns_{st.session_state.session_id}"
                vectorstore = LangchainPinecone(index, embedding_function, namespace, text_key="page_content")
                vectorstore.add_documents(chunks)
                save_document_metadata(doc_name, doc_type)
            except Exception as e:
                st.error(f"Error adding documents to existing DB: {str(e)}")
                # Fallback to re-initializing if adding fails
                vector_db = initialize_vector_db(chunks)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    save_document_metadata(doc_name, doc_type)

    except Exception as e:
        st.error(f"Document processing error: {str(e)}")

def load_doc_to_db(uploaded_files):
    """Load documents to vector database"""
    if uploaded_files:
        for doc_file in uploaded_files:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                    st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
                    break

                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(doc_file.getvalue())
                        tmp_path = tmp_file.name

                    loader = None
                    if doc_file.type == "application/pdf":
                        loader = PyPDFLoader(tmp_path)
                    elif doc_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(tmp_path)
                    elif doc_file.type in ["text/plain", "text/markdown"]:
                        loader = TextLoader(tmp_path)

                    if loader:
                        docs = loader.load()
                        process_documents(docs, doc_file.name, doc_file.type)
                        st.session_state.rag_sources.append(doc_file.name)

                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"Error loading {doc_file.name}: {str(e)}")

        if uploaded_files:
            st.success(f"Documents loaded successfully!")
            st.rerun() # Rerun to update the displayed sources

def load_url_to_db(url):
    """Load URL content to vector database"""
    if url and url not in st.session_state.rag_sources:
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
            return

        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            process_documents(docs, url, "url")
            st.session_state.rag_sources.append(url)
            st.success(f"URL content loaded successfully!")
            st.rerun() # Rerun to update the displayed sources
        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")

def initialize_documents():
    """Load document metadata from Pinecone on startup"""
    load_persisted_documents()

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
    response_message = ""

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