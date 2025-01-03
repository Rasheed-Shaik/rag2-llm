import streamlit as st
import os
import dotenv
import time
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec # Import Pinecone class and ServerlessSpec
import tempfile

from typing import List
from pathlib import Path
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings


from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    TextLoader,
)

dotenv.load_dotenv()

DB_DOCS_LIMIT = 10

def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

# --- Indexing Phase ---

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        st.write(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")
                    st.write(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")

def initialize_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name):
    """
    Initialize Pinecone vector database.
    """
    try:
        # Initialize Pinecone connection
        st.write("Attempting to initialize Pinecone connection...")
        pinecone_client = PineconeClient(api_key=pinecone_api_key) # New way, but not used directly
        st.write("Pinecone connection initialized successfully.")

        # Initialize embedding function
        # model_name = "Alibaba-NLP/gte-large-en-v1.5" # Old model
        model_name = "sentence-transformers/all-MiniLM-L6-v2" # New model
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True}
        )
        
        # Get the dimension of the embedding model
        embedding_dimension = len(embedding_function.embed_query("test"))

        # Check if the index exists
        st.write(f"Checking if Pinecone index '{pinecone_index_name}' exists...")
        index_names_method = pinecone_client.list_indexes() # Get the method object
        index_names = index_names_method().names if hasattr(index_names_method, '__call__') else [] # Call the method and access names
        st.write(f"Index names: {index_names}") # Debugging output
        if pinecone_index_name not in index_names:
            # Create a new index with the correct dimension
            st.write(f"Pinecone index '{pinecone_index_name}' does not exist. Creating it...")
            pinecone_client.create_index(
                name=pinecone_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=pinecone_environment
                )
            )
            st.write(f"Pinecone index '{pinecone_index_name}' created successfully.")
        else:
            st.write(f"Pinecone index '{pinecone_index_name}' already exists.")

        # Initialize Pinecone vector store
        st.write(f"Attempting to load Pinecone index: {pinecone_index_name}")
        vector_db = Pinecone.from_existing_index(
            index_name=pinecone_index_name,
            embedding=embedding_function,
        )
        st.write(f"Pinecone index '{pinecone_index_name}' loaded successfully.")
        return vector_db
        
    except Exception as e:
        error_msg = f"Pinecone database initialization failed: {str(e)}"
        st.write(f"Pinecone database initialization failed: {str(e)}")
        st.write(f"Detailed error: {error_msg}")
        st.error("Unable to process documents. Please try again.")
        return None


def _split_and_load_docs(docs: List[Document]):
    """
    Split documents and load them into the vector database
    """
    if not docs:
        return
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=1000
        )
        
        document_chunks = text_splitter.split_documents(docs)
        
        if not document_chunks:
            st.warning("No content was extracted from the documents.")
            return
            
        if "vector_db" not in st.session_state or st.session_state.vector_db is None:
            st.error("Pinecone vector database not initialized. Please check your API key and environment.")
            return
        else:
            try:
                st.session_state.vector_db.add_documents(document_chunks)
                st.write(f"Added {len(document_chunks)} document chunks to Pinecone.")
            except Exception as add_error:
                st.write(f"Error adding documents: {add_error}")
                st.error("Error adding documents to Pinecone. Please try again.")
            
    except Exception as e:
        st.write(f"Document processing error: {str(e)}")
        st.error("Error processing documents. Please try again.")
# --- Retrieval Augmented Generation (RAG) Phase ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but now always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})