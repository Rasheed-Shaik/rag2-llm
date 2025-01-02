# rag_methods.py
import streamlit as st
import os
import tempfile
from typing import List
from pathlib import Path
from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage
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

@st.cache_resource()
def get_pinecone_index():
    pinecone_client = Pinecone(
        api_key=st.secrets.get("PINECONE_API_KEY")
    )
    # Create index if it doesn't exist (important for the first time)
    existing_indexes = pinecone_client.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        st.info(f"Creating Pinecone index '{INDEX_NAME}'...")
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=384,  # dimension for BAAI text embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        st.success(f"Pinecone index '{INDEX_NAME}' created successfully.")
    return pinecone_client.Index(INDEX_NAME)

def initialize_pinecone():
    """Initialize Pinecone client using the new Pinecone class"""
    st.write("initialize_pinecone: START")
    try:
        index = get_pinecone_index()
        st.write("initialize_pinecone: END - Index exists or was created")
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone client or creating index: {e}")
        st.write("initialize_pinecone: END - Error")
        return None  # Return None if initialization or creation fails

def get_embedding_function():
    """Get the embedding function"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

def save_document_metadata(doc_name: str, doc_type: str):
    """Save document metadata to Pinecone"""
    st.write(f"save_document_metadata: START - doc_name: {doc_name}, doc_type: {doc_type}")
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

        vectorstore = LangchainPinecone(index, embedding_function, METADATA_NAMESPACE) # Removed text_key
        vectorstore.add_documents([metadata_doc])
        st.write("save_document_metadata: END - Metadata saved")

    except Exception as e:
        st.error(f"Error saving document metadata: {str(e)}")
        st.write("save_document_metadata: END - Error")

def get_metadata_store():
    """Get or initialize the metadata store."""
    if "metadata_store" not in st.session_state:
        st.write("get_metadata_store: metadata_store not in session_state, initializing")
        try:
            embedding_function = get_embedding_function()
            index = initialize_pinecone()

            st.session_state.metadata_store = LangchainPinecone(
                index,
                embedding=embedding_function,
                namespace=METADATA_NAMESPACE,
                text_key="page_content"
            )
            st.write("get_metadata_store: metadata_store initialized")
        except Exception as e:
            st.error(f"Error initializing metadata store: {str(e)}")
            st.write("get_metadata_store: Error initializing")
            return None
    else:
        st.write("get_metadata_store: metadata_store found in session_state")
    return st.session_state.metadata_store

def load_persisted_documents():
    """Load document metadata from Pinecone"""
    st.write("load_persisted_documents: START")
    metadata_store = get_metadata_store()
    if not metadata_store:
        st.write("load_persisted_documents: END - No metadata_store")
        return

    try:
        # Query all documents for the current session
        results = metadata_store.similarity_search(
            "document metadata",
            k=100, # Fetch a reasonable number of results
            filter={"session_id": st.session_state.session_id}
        )
        st.write(f"load_persisted_documents: Retrieved {len(results)} metadata entries")

        # Extract document names from metadata
        for result in results:
            try:
                metadata = json.loads(result.page_content)
                if metadata["name"] not in st.session_state.rag_sources:
                    st.session_state.rag_sources.append(metadata["name"])
                    st.write(f"load_persisted_documents: Added source: {metadata['name']}")
            except json.JSONDecodeError:
                st.error(f"Error decoding metadata: {result.page_content}")
                st.write(f"load_persisted_documents: Error decoding metadata: {result.page_content}")

        st.write("load_persisted_documents: END - Sources updated")

    except Exception as e:
        st.error(f"Error loading persisted documents: {str(e)}")
        st.write("load_persisted_documents: END - Error")

def initialize_documents():
    """Load document metadata from Pinecone on startup"""
    st.write("initialize_documents: START")
    load_persisted_documents()
    st.write("initialize_documents: END")

def initialize_vector_db(docs: List[Document]) -> LangchainPinecone:
    """Initialize vector database with provided documents"""
    st.write("initialize_vector_db: START")
    try:
        embedding_function = get_embedding_function()
        index = initialize_pinecone()  # Get the Pinecone Index object

        st.write(f"initialize_vector_db: Value of index after initialize_pinecone: {index}")

        if index is None:
            st.error("Failed to initialize Pinecone index.")
            st.write("initialize_vector_db: END - Pinecone initialization failed")
            return None

        st.write("initialize_vector_db: Index is not None, proceeding with LangchainPinecone")
        st.write(f"initialize_vector_db: Type of index: {type(index)}")

        vector_db = LangchainPinecone(
            index=index,  # Explicitly pass the index here
            embedding=embedding_function,
            namespace=f"ns_{st.session_state.session_id}",
            text_key="page_content",
        )

        st.write("initialize_vector_db: LangchainPinecone object created")
        vector_db.add_documents(documents=docs) # Add documents separately

        st.write("initialize_vector_db: END - Vector DB initialized")
        return vector_db
    except Exception as e:
        st.error(f"Vector DB initialization failed: {str(e)}")
        st.write("initialize_vector_db: END - Error")
        return None

# rag_methods.py
def process_documents(docs: List[Document], doc_name: str, doc_type: str) -> None:
    """Process and load documents into vector database"""
    st.write(f"process_documents: START - doc_name: {doc_name}, doc_type: {doc_type}, num_docs: {len(docs)}")
    if not docs:
        st.write("process_documents: END - No documents to process")
        return

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(docs)
        if not chunks:
            st.warning("No content extracted from documents.")
            st.write("process_documents: END - No content extracted")
            return

        if st.session_state.vector_db is None:
            st.write("process_documents: Initializing new vector DB")
            vector_db = initialize_vector_db(chunks)
            if vector_db:
                st.session_state.vector_db = vector_db
                save_document_metadata(doc_name, doc_type)
                st.write("process_documents: New vector DB initialized and set in session state.")
            else:
                st.write("process_documents: Failed to initialize new vector DB.")
        else:
            st.write("process_documents: Adding to existing vector DB")
            try:
                st.session_state.vector_db.add_documents(chunks)  # Use the existing vector_db object
                save_document_metadata(doc_name, doc_type)
                st.write("process_documents: Added documents to existing vector DB.")
            except Exception as e:
                st.error(f"Error adding documents to existing DB: {str(e)}")
                st.write(f"process_documents: Error adding to existing DB: {str(e)}, attempting re-initialization")
                # Fallback to re-initializing if adding fails (consider why this might be needed)
                vector_db = initialize_vector_db(chunks)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    save_document_metadata(doc_name, doc_type)
                    st.write("process_documents: Re-initialized vector DB after error.")
                else:
                    st.write("process_documents: Failed to re-initialize vector DB.")

        st.write("process_documents: END - Documents processed")

    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        st.write(f"process_documents: END - Error: {str(e)}")

def load_doc_to_db(uploaded_files):
    """Load documents to vector database"""
    st.write("load_doc_to_db: START")
    if uploaded_files:
        for doc_file in uploaded_files:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                    st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
                    st.write("load_doc_to_db: END - Document limit reached")
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
                        st.write(f"load_doc_to_db: Loaded document: {doc_file.name}")

                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"Error loading {doc_file.name}: {str(e)}")
                    st.write(f"load_doc_to_db: Error loading {doc_file.name}: {str(e)}")

        if uploaded_files:
            st.success(f"Documents loaded successfully!")
            st.write("load_doc_to_db: END - Documents loaded")
    else:
        st.write("load_doc_to_db: END - No files uploaded")

def load_url_to_db(url):
    """Load URL content to vector database"""
    st.write(f"load_url_to_db: START - URL: {url}")
    if url and url not in st.session_state.rag_sources:
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Document limit ({DB_DOCS_LIMIT}) reached.")
            st.write("load_url_to_db: END - Document limit reached")
            return

        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            process_documents(docs, url, "url")
            st.session_state.rag_sources.append(url)
            st.success(f"URL content loaded successfully!")
            st.write("load_url_to_db: END - URL loaded")
        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")
            st.write(f"load_url_to_db: END - Error loading URL: {str(e)}")
    else:
        st.write("load_url_to_db: END - URL already loaded or empty")

def initialize_documents():
    """Load document metadata from Pinecone on startup"""
    st.write("initialize_documents: START")
    load_persisted_documents()
    st.write("initialize_documents: END")

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

def stream_llm_response(llm_stream, messages: List[BaseMessage]):
    """Stream LLM response without RAG"""
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content
    st.session_state.messages.append(AIMessage(content=response_message))

def stream_llm_rag_response(llm_stream, messages: List[BaseMessage]):
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