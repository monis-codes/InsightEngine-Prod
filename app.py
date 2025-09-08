import os
import hashlib

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

from rag_logic import process_document, get_answer


load_dotenv()

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("Document Q&A (RAG)")

# Sidebar controls
with st.sidebar:
    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"]) 
    process_clicked = st.button("Process Document", type="primary")
    st.caption("Supported: PDF")


# Session state
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def _get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    if not api_key or not index_name:
        st.error("PINECONE_API_KEY and PINECONE_INDEX must be set in environment.")
        st.stop()
    pc = Pinecone(api_key=api_key)
    try:
        index = pc.Index(index_name)
    except Exception as e:
        st.error(f"Failed to access Pinecone index '{index_name}': {e}")
        st.stop()
    return index


index = _get_pinecone_index()


# Processing flow
if process_clicked:
    if not uploaded_file:
        st.warning("Please upload a document first.")
    else:
        file_bytes = uploaded_file.getvalue()
        doc_id = hashlib.md5(file_bytes).hexdigest()
        with st.spinner("Processing document..."):
            try:
                ok = process_document(index, file_bytes, doc_id)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                ok = False
        if ok:
            st.session_state.doc_id = doc_id
            st.success("Document processed and indexed.")


# Chat history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)


# Chat input
prompt = st.chat_input("Ask a question about the processed document…")
if prompt:
    if not st.session_state.doc_id:
        st.warning("No active document. Please upload and process a document first.")
    else:
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                answer = get_answer(index, prompt, st.session_state.doc_id)
            except Exception as e:
                answer = f"Error: {e}"
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))



