result = rag_chain.invoke({"messages": messages[:-1], "input": messages[-1].content})
answer = result.get("answer", "")
source_documents = result.get("source_documents", [])

# ... (streaming the answer) ...

if source_documents:
    with st.expander("Sources"):
        for doc in source_documents:
            st.markdown(f"> `{doc.metadata.get('source', 'unknown')}`")