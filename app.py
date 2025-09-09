import os
import hashlib
import time
import re
from secrets import randbelow

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

from rag_logic import process_document, get_answer


load_dotenv()

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("Document Q&A (RAG)")
st.markdown(
    "This is my demo for RAG pipeline project. This is the light v1 version. "
    "For a more optimized and accurate version, refer to this repo "
    "[rxHackathon](https://github.com/monis-codes/rxHackathon). "
    "Thank you for coming here. This is open source and everyone is welcome to "
    "contribute in this repo "
    "[rag-Deploy](https://github.com/monis-codes/rag-Deploy)."
)

# Sidebar controls (moved below after function definitions)


# Session state
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "captcha" not in st.session_state:
    st.session_state.captcha = {
        "a": None,
        "b": None,
        "verified_at": None,
    }

# Security/config constants
CAPTCHA_EXPIRY_SECONDS = 5 * 60  # require re-verification periodically
MAX_PROMPT_CHARS = 500


def _generate_captcha():
    a = randbelow(9) + 1
    b = randbelow(9) + 1
    st.session_state.captcha["a"] = a
    st.session_state.captcha["b"] = b
    st.session_state.captcha["verified_at"] = None


def _captcha_is_valid() -> bool:
    verified_at = st.session_state.captcha.get("verified_at")
    if not verified_at:
        return False
    return (time.time() - verified_at) <= CAPTCHA_EXPIRY_SECONDS


def _show_captcha_sidebar():
    st.subheader("Verification")
    if st.session_state.captcha["a"] is None or st.session_state.captcha["b"] is None:
        _generate_captcha()
    a = st.session_state.captcha["a"]
    b = st.session_state.captcha["b"]

    cols = st.columns([3, 1])
    with cols[0]:
        answer_input = st.text_input(
            f"What is {a} + {b}?",
            key="captcha_input",
            placeholder="Enter answer",
        )
    with cols[1]:
        if st.button("New"):
            _generate_captcha()

    verified = _captcha_is_valid()
    if st.button("Verify"):
        try:
            if answer_input is not None and int(answer_input.strip()) == (a + b):
                st.session_state.captcha["verified_at"] = time.time()
                verified = True
            else:
                st.warning("Incorrect CAPTCHA answer. Try again or click New.")
        except Exception:
            st.warning("Please enter a numeric answer.")

    if verified:
        remaining = max(0, CAPTCHA_EXPIRY_SECONDS - int(time.time() - st.session_state.captcha["verified_at"]))
        st.success(f"CAPTCHA verified. Expires in ~{remaining}s")
    else:
        st.info("Please verify the CAPTCHA to enable actions.")


def _contains_malicious_pattern(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"<\s*script\b",
        r"\b(onerror|onload|onclick)\s*=",
        r"\b(drop|delete|insert|update|union\s+select)\b",
        r"\b(curl|wget|powershell|bash|sh\b|cmd\.exe)\b",
        r"\b--\b|/\*|\*/",
    ]
    lowered = str(text).lower()
    for pattern in patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True
    return False


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


# Sidebar controls
with st.sidebar:
    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"]) 
    process_clicked = st.button("Process Document", type="primary")
    st.caption("Supported: PDF")
    st.divider()
    _show_captcha_sidebar()


# Processing flow
if process_clicked:
    if not uploaded_file:
        st.warning("Please upload a document first.")
    else:
        if not _captcha_is_valid():
            st.warning("Action blocked. Please verify the CAPTCHA in the sidebar.")
            st.stop()
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
prompt = st.chat_input(f"Ask a question (max {MAX_PROMPT_CHARS} chars)…")
if prompt:
    if not st.session_state.doc_id:
        st.warning("No active document. Please upload and process a document first.")
    else:
        if not _captcha_is_valid():
            st.warning("Please re-verify the CAPTCHA to continue chatting.")
        elif len(prompt) > MAX_PROMPT_CHARS:
            st.warning(f"Your message exceeds the {MAX_PROMPT_CHARS}-character limit.")
        elif _contains_malicious_pattern(prompt):
            st.warning("Your message was blocked due to suspicious content. Please modify and try again.")
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



