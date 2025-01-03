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
    st.write(f"Pinecone index '{pinecone_index_name}' creation initiated.")
    
    # Wait for the index to be ready
    while True:
        index_description = pinecone_client.describe_index(pinecone_index_name)
        if index_description.status.ready:
            st.write(f"Pinecone index '{pinecone_index_name}' is ready.")
            break
        else:
            st.write(f"Waiting for Pinecone index '{pinecone_index_name}' to be ready...")
            time.sleep(5) # Wait for 5 seconds before checking again