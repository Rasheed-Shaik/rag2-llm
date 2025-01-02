# minimal_app.py
import streamlit as st
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

INDEX_NAME = "langchain-rag"
NAMESPACE = "test_namespace"

@st.cache_resource()
def initialize_pinecone_client():
    return Pinecone(api_key=st.secrets.get("PINECONE_API_KEY"))

@st.cache_resource()
def get_pinecone_index(client):
    return client.Index(INDEX_NAME)

@st.cache_resource()
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

st.title("Minimal LangchainPinecone Test")

pinecone_client = initialize_pinecone_client()
index = get_pinecone_index(pinecone_client)
embeddings = get_embedding_function()

try:
    vectorstore = LangchainPinecone(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
        text_key="text"
    )
    st.success("LangchainPinecone initialized successfully!")
    st.write(f"Vectorstore: {vectorstore}")
except Exception as e:
    st.error(f"Error initializing LangchainPinecone: {e}")