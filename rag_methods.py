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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chromadb.config import Settings
from langchain_google_genai import ChatGoogleGenerativeAI

DB_DOCS_LIMIT = 10
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def initialize_llm(model, google_api_key):
    """Initialize the LLM."""
    return ChatGoogleGenerativeAI(
        model=model.split("/")[-1],
        google_api_key=google_api_key,
        temperature=0,
        streaming=True
    )



def stream_llm_response(llm_stream, messages):
    """Stream LLM response without RAG"""
    response_message = ""
    for chunk in llm_stream.stream(messages):
        # Inspect the chunk structure (for debugging - you can remove this later)
        print(f"Chunk received: {chunk}")

        text_content = None
        if hasattr(chunk, 'content'):
            if isinstance(chunk.content, list):
                text_content = "".join(chunk.content)
            else:
                text_content = chunk.content
        elif isinstance(chunk, dict) and 'text' in chunk:
            text_content = chunk['text']
        elif isinstance(chunk, dict) and 'candidates' in chunk and chunk['candidates']:
            # Assuming 'candidates' is a list of potential responses
            best_candidate = chunk['candidates'][0]  # Or some logic to choose the best
            if 'content' in best_candidate and 'parts' in best_candidate['content']:
                text_content = "".join([part['text'] for part in best_candidate['content']['parts']])
        # Add more checks here if you find other structures in the raw chunks

        if text_content:
            response_message += text_content
            yield text_content

    st.session_state.messages.append({"role": "assistant", "content": response_message})

def initialize_vector_db(docs: List[Document]) -> Chroma:
    """Initialize vector database with provided documents"""
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        temp_dir = tempfile.mkdtemp()
        chroma_settings = Settings(
            is_persistent=True,
            persist_directory=temp_dir,
            anonymized_telemetry=False
        )

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            collection_name=f"collection_{st.session_state.session_id}",
            persist_directory=temp_dir,
            client_settings=chroma_settings
        )

        return vector_db
    except Exception as e:
        st.error(f"Vector DB initialization failed: {e}")
        return None

def process_documents(docs: List[Document]) -> None:
    """Process and load documents into vector database"""
    if not docs:
        st.warning("No documents to process.")
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
            except Exception as e:
                st.error(f"Error adding documents to existing vector DB: {e}")
                vector_db = initialize_vector_db(chunks)
                if vector_db:
                    st.session_state.vector_db = vector_db

    except Exception as e:
        st.error(f"Document processing error: {e}")


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

                        loaders = {
                            "application/pdf": PyPDFLoader,
                            ".docx": Docx2txtLoader,
                            "text/plain": TextLoader,
                            "text/markdown": TextLoader
                        }

                        loader_class = loaders.get(doc_file.type, None)
                        if not loader_class:
                          for key in loaders.keys():
                              if doc_file.name.endswith(key):
                                  loader_class = loaders.get(key)
                                  break

                        if loader_class:
                            loader = loader_class(file_path)
                            loaded_docs = loader.load()
                            docs.extend(loaded_docs)
                            st.session_state.rag_sources.append(doc_file.name)

                except Exception as e:
                    st.error(f"Error loading {doc_file.name}: {e}")
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
                st.error(f"Error loading URL: {e}")

def get_rag_chain(llm):
    """Create RAG chain for conversational retrieval"""
    if st.session_state.vector_db is None:
        st.error("Vector database is not initialized.")
        return None

    retriever = st.session_state.vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    context_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant, use the context to answer the user's question."
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Generate a search query based on our conversation, focusing on recent messages.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, context_prompt)

    response_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Answer based on the context and your knowledge. Context: {context}"
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}")
    ])

    return create_retrieval_chain(
        retriever_chain,
        create_stuff_documents_chain(llm, response_prompt)
    )

def stream_llm_rag_response(llm, messages):
    """Stream RAG-enhanced LLM response"""
    rag_chain = get_rag_chain(llm)
    if rag_chain is None:
        return

    response_message = "🔍 "
    for chunk in rag_chain.stream({
        "messages": messages[:-1],
        "input": messages[-1].content
    }):
        if isinstance(chunk, dict) and "text" in chunk:
            if isinstance(chunk["text"], list):
                 response_message += "".join(chunk["text"])
                 yield "".join(chunk["text"])
            else:
                response_message += chunk["text"]
                yield chunk["text"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_message
    })