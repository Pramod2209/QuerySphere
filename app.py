import streamlit as st
from sentence_transformers import SentenceTransformer
from backend import process_query
import json
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- API Key Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not set!")
    st.stop()

# --- Page Setup ---
st.set_page_config(
    page_title="QUERY SPHERE",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --accent: #4776E6;
        --dark: #1a1a2e;
        --light: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: var(--dark);
    }
    .stButton>button {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(to right, var(--secondary), var(--primary));
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 0.75rem 1rem;
        border: 1px solid #ddd;
    }
    .stFileUploader>div>div>div>div {
        border-radius: 20px;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.8);
        border: 2px dashed var(--primary);
    }
    .upload-option {
        margin-bottom: 1rem;
    }
    .or-divider {
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        color: var(--secondary);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
        <h2 style="color:white;">QUERY SPHERE 🚀</h2>
        <p style="color:#aaa;">AI Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How to use:")
    st.markdown("1. Upload a document or paste URL")
    st.markdown("2. Ask your question")
    st.markdown("3. Get instant insights")
    st.markdown("---")
    st.markdown("### Supported formats:")
    st.markdown("- PDF, DOCX, EML files")
    st.markdown("- Public document URLs")

# --- Session State ---
if "result" not in st.session_state:
    st.session_state.result = None

# --- Model Loading ---
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# --- Main Content ---
st.title("QUERY SPHERE")
st.markdown("Analyze documents with AI")

# --- Document Input Section ---
upload_option = st.radio(
    "Choose input method:",
    ["Upload a file", "Paste document URL"],
    horizontal=True
)

document_input = None
if upload_option == "Upload a file":
    document_input = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "eml"],
        label_visibility="collapsed"
    )
else:
    document_input = st.text_input(
        "Paste document URL",
        placeholder="https://example.com/document.pdf",
        label_visibility="collapsed"
    )

# --- Query Input ---
query = st.text_input(
    "Enter your question",
    placeholder="What are the key points in this document?",
    label_visibility="collapsed"
)

# --- Process Button ---
if st.button("Analyze Document", use_container_width=True):
    if not document_input:
        st.error("Please provide a document or URL")
    elif not query:
        st.error("Please enter a question")
    else:
        with st.spinner("Processing..."):
            try:
                embedding_model = load_embedding_model()
                st.session_state.result = process_query(
                    GROQ_API_KEY,
                    document_input,
                    query,
                    embedding_model
                )
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# --- Results Display ---
if st.session_state.result:
    result = st.session_state.result
    st.markdown("---")
    st.subheader("Analysis Results")

    if "error" in result:
        st.error(result["error"])
    else:
        st.success("Analysis complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Relevant Clause**")
            st.info(result.get("relevant_clause", "N/A"))

        with col2:
            st.markdown("**Source Reference**")
            st.info(result.get("source_reference", "N/A"))

        st.markdown("**Explanation**")
        st.write(result.get("explanation", "No explanation provided."))

        with st.expander("View Raw JSON"):
            st.json(result)

        st.download_button(
            label="Download Results",
            data=json.dumps(result, indent=2),
            file_name="analysis_results.json",
            mime="application/json"
        )

        if st.button("New Analysis", use_container_width=True):
            st.session_state.result = None
            st.rerun()
