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
    try:
        index = get_pinecone_index()
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone client or creating index: {e}")
        return None

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

def save_document_metadata(doc_name: str, doc_type: str):
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
            metadata={"source": doc_name}
        )

        vectorstore = LangchainPinecone(index, embedding_function, METADATA_NAMESPACE)
        vectorstore.add_documents([metadata_doc])

    except Exception as e:
        st.error(f"Error saving document metadata: {str(e)}")

def get_metadata_store():
    print("Entering get_metadata_store()") # Log entry
    if "metadata_store" not in st.session_state:
        print("metadata_store not in session_state, initializing...")
        try:
            embedding_function = get_embedding_function()
            index = initialize_pinecone()

            st.session_state.metadata_store = LangchainPinecone(
                index,
                embedding=embedding_function,
                namespace=METADATA_NAMESPACE,
                text_key="page_content"
            )
            print("metadata_store initialized successfully.")
        except Exception as e:
            st.error(f"Error initializing metadata store: {str(e)}")
            print(f"Error details in get_metadata_store: {str(e)}")
            return None
    else:
        print("metadata_store found in session_state.")
    print("Exiting get_metadata_store()") # Log exit
    return st.session_state.metadata_store

def load_persisted_documents():
    print("Entering load_persisted_documents()")  # Logging entry
    metadata_store = get_metadata_store()
    if not metadata_store:
        print("Metadata store is not initialized.")
        return

    try:
        print(f"Current session_id: {st.session_state.session_id}") # Log session ID
        # TEMPORARY: Removing session_id filter for testing
        results = metadata_store.similarity_search(
            "document metadata",
            k=100,
            # filter={"session_id": st.session_state.session_id} # Original filtering
        )
        print(f"Number of metadata results found: {len(results)}") # Log results

        for result in results:
            try:
                metadata = json.loads(result.page_content)
                print(f"Loaded metadata: {metadata}") # Log loaded metadata
                if metadata["name"] not in st.session_state.rag_sources:
                    st.session_state.rag_sources.append(metadata["name"])
            except json.JSONDecodeError:
                st.error(f"Error decoding metadata: {result.page_content}")

        print(f"rag_sources after loading: {st.session_state.rag_sources}") # Log final rag_sources

    except Exception as e:
        st.error(f"Error loading persisted documents: {str(e)}")
        print(f"Error details: {str(e)}") # Log error details
    print("Exiting load_persisted_documents()") # Log exit

def initialize_documents():
    print("Entering initialize_documents()") # Log entry
    load_persisted_documents()
    print("Exiting initialize_documents()") # Log exit

def initialize_vector_db(docs: List[Document]) -> LangchainPinecone:
    try:
        embedding_function = get_embedding_function()
        index = initialize_pinecone()

        if index is None:
            st.error("Failed to initialize Pinecone index.")
            return None

        namespace = f"ns_{st.session_state.session_id}"
        print(f"Initializing vector DB with namespace: {namespace}") # Log namespace
        vector_db = LangchainPinecone(
            index=index,
            embedding=embedding_function,
            namespace=namespace,
            text_key="page_content",
        )
        vector_db.add_documents(documents=docs)
        return vector_db
    except Exception as e:
        st.error(f"Vector DB initialization failed: {str(e)}")
        return None

# rag_methods.py
def process_documents(docs: List[Document], doc_name: str, doc_type: str) -> None:
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
                save_document_metadata(doc_name, doc_type)
            else:
                st.error("Failed to initialize new vector DB.")
        else:
            try:
                st.session_state.vector_db.add_documents(chunks)
                save_document_metadata(doc_name, doc_type)
            except Exception as e:
                st.error(f"Error adding documents to existing DB: {str(e)}")
                vector_db = initialize_vector_db(chunks)
                if vector_db:
                    save_document_metadata(doc_name, doc_type)
                else:
                    st.error("Failed to re-initialize vector DB.")

    except Exception as e:
        st.error(f"Document processing error: {str(e)}")

def load_doc_to_db(uploaded_files):
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

def load_url_to_db(url):
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
        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")

def initialize_documents():
    print("Entering initialize_documents()") # Log entry
    load_persisted_documents()
    print("Exiting initialize_documents()") # Log exit

def get_rag_chain(llm):
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
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content
    st.session_state.messages.append(AIMessage(content=response_message))

def stream_llm_rag_response(llm_stream, messages: List[BaseMessage]):
    rag_chain = get_rag_chain(llm_stream)
    response_message = ""

    for chunk in rag_chain.pick("answer").stream({
        "messages": messages[:-1],
        "input": messages[-1].content
    }):
        response_message += chunk
        yield chunk

    st.session_state.messages.append(AIMessage(content=response_message))