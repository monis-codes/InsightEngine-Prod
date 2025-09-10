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

# Enhanced CSS for improved UI/UX
st.markdown("""
<style>
    /* Enhanced Color Palette & Theming */
    :root {
        --accent-blue: #007BFF;
        --accent-blue-hover: #0056b3;
        --accent-blue-light: rgba(0, 123, 255, 0.1);
        --accent-blue-border: rgba(0, 123, 255, 0.3);
        --success-green: #28a745;
        --error-red: #dc3545;
        --warning-orange: #ffc107;
        --text-white: #ffffff;
        --text-light-grey: #e0e0e0;
        --text-body-grey: #b0b0b0;
        --text-placeholder-grey: #808080;
        --background-dark: #0e1117;
        --background-card: rgba(255, 255, 255, 0.05);
    }
    
    /* Enhanced main title styling - Left aligned */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: var(--text-white) !important;
        margin-bottom: 0.5rem !important;
        text-align: left !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .title-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: fadeInDown 1s ease-out;
        justify-content: flex-start;
    }
    
    .lightbulb-icon {
        font-size: 3.5rem;
        color: var(--accent-blue);
        filter: drop-shadow(0 0 15px var(--accent-blue));
        animation: pulse 2s infinite;
    }
    
    /* Enhanced introduction text styling - Left aligned */
    .intro-text {
        color: var(--text-body-grey) !important;
        font-size: 1.2rem !important;
        text-align: left !important;
        margin-bottom: 2rem !important;
        line-height: 1.7 !important;
        max-width: 100%;
    }
    
    /* Welcome container styling */
    .welcome-container {
        background-color: var(--background-card) !important;
        border: 2px solid var(--accent-blue-border) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin-bottom: 3rem !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    .welcome-header {
        margin-bottom: 1.5rem !important;
    }
    
    .cta-text {
        color: var(--accent-blue) !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced sidebar styling */
    .sidebar-section {
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: var(--background-card);
        border-radius: 8px;
        border: 1px solid var(--accent-blue-border);
    }
    
    .section-header {
        color: var(--text-white) !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .section-icon {
        font-size: 1.3rem;
        filter: drop-shadow(0 0 5px var(--accent-blue));
    }
    
    /* Enhanced file uploader styling */
    .stFileUploader > div > div {
        border: 2px dashed var(--accent-blue) !important;
        border-radius: 12px !important;
        background-color: var(--accent-blue-light) !important;
        transition: all 0.3s ease !important;
        padding: 2rem !important;
        text-align: center !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--accent-blue-hover) !important;
        background-color: rgba(0, 123, 255, 0.15) !important;
        transform: scale(1.02) !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-hover)) !important;
        color: var(--text-white) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--accent-blue-hover), #004085) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 123, 255, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.3) !important;
    }
    
    /* Enhanced chat input styling - Left aligned */
    .chat-input-container {
        background-color: rgba(0, 123, 255, 0.08) !important;
        border: 2px solid var(--accent-blue) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2);
    }
    
    .chat-header {
        color: var(--text-white) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-align: left !important;
        margin-bottom: 1.5rem !important;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .chat-icon {
        font-size: 1.5rem;
        color: var(--accent-blue);
        filter: drop-shadow(0 0 8px var(--accent-blue));
    }
    
    /* Enhanced repository links styling */
    .repo-links {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        background-color: rgba(0, 0, 0, 0.8);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: var(--text-body-grey);
        z-index: 1000;
        border: 1px solid var(--accent-blue-border);
        backdrop-filter: blur(10px);
    }
    
    .repo-links a {
        color: var(--accent-blue) !important;
        text-decoration: none !important;
        margin: 0 0.5rem;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .repo-links a:hover {
        color: var(--accent-blue-hover) !important;
        text-decoration: underline !important;
    }
    
    /* Enhanced loading animation */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, var(--accent-blue-light), rgba(0, 123, 255, 0.05));
        border-radius: 12px;
        margin: 2rem 0;
        border: 1px solid var(--accent-blue-border);
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2);
    }
    
    .loading-text {
        color: var(--accent-blue) !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    
    .loading-spinner {
        width: 24px;
        height: 24px;
        border: 3px solid var(--accent-blue-border);
        border-top: 3px solid var(--accent-blue);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    /* Enhanced message styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.15), rgba(40, 167, 69, 0.05)) !important;
        border: 2px solid var(--success-green) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.15), rgba(220, 53, 69, 0.05)) !important;
        border: 2px solid var(--error-red) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(220, 53, 69, 0.2) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15), rgba(255, 193, 7, 0.05)) !important;
        border: 2px solid var(--warning-orange) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(255, 193, 7, 0.2) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--accent-blue-light), rgba(0, 123, 255, 0.05)) !important;
        border: 2px solid var(--accent-blue) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2) !important;
    }
    
    /* Enhanced chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid var(--accent-blue-border) !important;
    }
    
    /* Output area styling */
    .output-container {
        background-color: var(--background-card) !important;
        border: 2px solid var(--accent-blue-border) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        min-height: 100px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem !important;
        }
        
        .lightbulb-icon {
            font-size: 2.5rem;
        }
        
        .repo-links {
            position: relative;
            bottom: auto;
            right: auto;
            margin-top: 2rem;
            text-align: center;
        }
        
        .chat-header {
            font-size: 1.4rem !important;
        }
        
        .cta-text {
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="InsightEngine", layout="wide")

# Title section
st.markdown("""
<div class="title-container">
    <span class="lightbulb-icon">💡</span>
    <h1 class="main-title">Insight Engine</h1>
</div>
""", unsafe_allow_html=True)

# Introduction text
st.markdown("""
<div class="intro-text">
    This is a lightweight, proof-of-concept version of a Retrieval-Augmented Generation (RAG) pipeline, designed to be computationally efficient for public deployment.
    To demonstrate the full scope of the architecture and our technical capabilities, the core project (unavailable for public demo due to the GPU-heavy requirements of PyTorch) utilizes advanced components, including a <b>TinyBERT cross-encoder</b> for more effective retrieval and a <b>HyDE-based query generator</b> for refined reranking.
    For a detailed look at the complete, optimized codebase, please refer to the <a href="https://github.com/monis-codes/rxHackathon" style="color: var(--accent-blue);">this</a> repository.
</div>
""", unsafe_allow_html=True)

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
    a = randbelow(20) + 1
    b = randbelow(20) + 1
    st.session_state.captcha["a"] = a
    st.session_state.captcha["b"] = b
    st.session_state.captcha["verified_at"] = None


def _captcha_is_valid() -> bool:
    verified_at = st.session_state.captcha.get("verified_at")
    if not verified_at:
        return False
    return (time.time() - verified_at) <= CAPTCHA_EXPIRY_SECONDS


def _show_captcha_sidebar():
    st.markdown("""
    <div class="sidebar-section">
        <div class="section-header">
            <span class="section-icon">✅</span>
            Verification
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.captcha["a"] is None or st.session_state.captcha["b"] is None:
        _generate_captcha()
    a = st.session_state.captcha["a"]
    b = st.session_state.captcha["b"]

    cols = st.columns([3, 1])
    with cols[0]:
        answer_input = st.text_input(
            f"🧮 What is {a} + {b}?",
            key="captcha_input",
            placeholder="Enter answer",
            help="Solve this simple math problem to verify you're human"
        )
    with cols[1]:
        if st.button("🔄 New"):
            _generate_captcha()

    verified = _captcha_is_valid()
    if st.button("✅ Verify"):
        try:
            if answer_input is not None and int(answer_input.strip()) == (a + b):
                st.session_state.captcha["verified_at"] = time.time()
                verified = True
            else:
                st.warning("❌ Incorrect CAPTCHA answer. Try again or click New.")
        except Exception:
            st.warning("⚠️ Please enter a numeric answer.")

    if verified:
        remaining = max(0, CAPTCHA_EXPIRY_SECONDS - int(time.time() - st.session_state.captcha["verified_at"]))
        st.success(f"✅ CAPTCHA verified. Expires in ~{remaining}s")
    else:
        st.info("🔒 Please verify the CAPTCHA to enable actions.")


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


# Enhanced sidebar controls
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <div class="section-header">
            <span class="section-icon">📁</span>
            Upload & Process
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📄 Upload a PDF Document", type=["pdf"], help="Drag and drop your PDF file here or click to browse") 
    process_clicked = st.button("🚀 Process Document", type="primary")
    
    st.info("📄 Only PDF files are supported. Maximum file size: 200MB")
    
    st.markdown("---")
    
    _show_captcha_sidebar()


# Enhanced processing flow with better feedback
if process_clicked:
    if not uploaded_file:
        st.warning("📄 Please upload a document first.")
    else:
        if not _captcha_is_valid():
            st.warning("🔒 Action blocked. Please verify the CAPTCHA in the sidebar.")
            st.stop()
        file_bytes = uploaded_file.getvalue()
        doc_id = hashlib.md5(file_bytes).hexdigest()
        
        # Enhanced loading with custom styling
        with st.spinner("🔄 Processing document and preparing knowledge base..."):
            try:
                ok = process_document(index, file_bytes, doc_id)
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                ok = False
        
        if ok:
            st.session_state.doc_id = doc_id
            st.success("✅ Document processed and indexed successfully!")


# Enhanced chat interface with left alignment
# st.markdown("""
# <div class="chat-input-container">
#     <div class="chat-header">
#         <span class="chat-icon">💬</span>
#         Ask Your Document
#     </div>
# </div>
# """, unsafe_allow_html=True)

# Chat history with enhanced styling
if st.session_state.messages:
    st.markdown("### 💭 Conversation History")
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)
else:
    st.markdown("""
    <div class="output-container">
        <div style="text-align: left; color: var(--text-body-grey); padding: 2rem;">
            <h3>🤖 Ready to Answer Your Questions!</h3>
            <p>Upload a document and start asking questions to get AI-powered insights.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced chat input with better loading feedback
prompt = st.chat_input(f"💭 Ask a question about your document (max {MAX_PROMPT_CHARS} chars)…")
if prompt:
    if not st.session_state.doc_id:
        st.warning("📄 No active document. Please upload and process a document first.")
    else:
        if not _captcha_is_valid():
            st.warning("🔒 Please re-verify the CAPTCHA to continue chatting.")
        elif len(prompt) > MAX_PROMPT_CHARS:
            st.warning(f"⚠️ Your message exceeds the {MAX_PROMPT_CHARS}-character limit.")
        elif _contains_malicious_pattern(prompt):
            st.warning("🚫 Your message was blocked due to suspicious content. Please modify and try again.")
        else:
            st.session_state.messages.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Enhanced loading with custom spinner
                with st.spinner("🧠 Generating insights..."):
                    try:
                        answer = get_answer(index, prompt, st.session_state.doc_id)
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                    st.markdown(answer)
            st.session_state.messages.append(("assistant", answer))

