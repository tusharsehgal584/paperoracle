import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import anthropic
import tempfile
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PaperOracle — Ask Your PDF",
    page_icon="🔮",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }

    .header-box {
        background: linear-gradient(135deg, #1a1025, #0d1b2a);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 28px;
        text-align: center;
    }
    .header-box h1 { color: #e6edf3; font-size: 2.2rem; margin: 0 0 6px 0; }
    .header-box p  { color: #8b949e; margin: 0; font-size: 0.96rem; }

    .concept-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #a371f7;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .concept-box h4 { color: #a371f7; margin: 0 0 6px 0; font-size: 0.92rem; }
    .concept-box p  { color: #8b949e; margin: 0; font-size: 0.86rem; line-height: 1.6; }

    .chat-user {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px 10px 2px 10px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #e6edf3;
        font-size: 0.93rem;
    }
    .chat-ai {
        background: #0d1117;
        border: 1px solid #6e40c9;
        border-radius: 2px 10px 10px 10px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #e6edf3;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .source-tag {
        background: #1a1025;
        border: 1px solid #6e40c9;
        color: #a371f7;
        border-radius: 4px;
        padding: 2px 9px;
        font-size: 0.74rem;
        display: inline-block;
        margin: 4px 4px 0 0;
    }
    .status-ready {
        background: #0f1d0f;
        border: 1px solid #238636;
        color: #3fb950;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        text-align: center;
        margin: 14px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6e40c9, #a371f7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #a371f7, #c9a7ff);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🔮 PaperOracle</h1>
    <p>Upload any PDF — ask questions in plain English — get sourced answers instantly</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — HOW IT WORKS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ How PaperOracle Works")
    st.markdown("""
    <div class="concept-box">
        <h4>Step 1 — Document Loading</h4>
        <p>Your PDF is read page by page using PyPDFLoader, extracting all raw text.</p>
    </div>
    <div class="concept-box">
        <h4>Step 2 — Text Chunking</h4>
        <p>Text is split into overlapping 500-char chunks so no context is lost at boundaries.</p>
    </div>
    <div class="concept-box">
        <h4>Step 3 — Embeddings</h4>
        <p>Each chunk is turned into a meaning-vector using a free local HuggingFace model.</p>
    </div>
    <div class="concept-box">
        <h4>Step 4 — FAISS Vector Store</h4>
        <p>All vectors are stored in FAISS — Meta's lightning-fast similarity search index.</p>
    </div>
    <div class="concept-box">
        <h4>Step 5 — RAG Retrieval</h4>
        <p>Your question is embedded and FAISS finds the 3 most relevant chunks as context.</p>
    </div>
    <div class="concept-box">
        <h4>Step 6 — Claude AI Answer</h4>
        <p>The chunks + question are sent to Claude (Anthropic) which generates a grounded answer.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Stack:** LangChain · FAISS · HuggingFace · Claude AI · Streamlit")
    st.markdown("**By:** [Tushar Sehgal](https://github.com/tusharsehgal584)")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("vectorstore", None),
    ("chat_history", []),
    ("doc_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# API KEY INPUT
# ─────────────────────────────────────────────
st.markdown("### 🔑 Anthropic API Key")
st.caption("Get your free key at console.anthropic.com — takes 1 minute")
anthropic_key = st.text_input(
    "Paste your Anthropic API key",
    type="password",
    placeholder="sk-ant-xxxxxxxxxxxxxxxxxxxx"
)

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown("### 📄 Upload Your PDF")
uploaded_file = st.file_uploader(
    "Any PDF — research papers, textbooks, resumes, reports, contracts",
    type=["pdf"]
)

# ─────────────────────────────────────────────
# CORE PIPELINE: Build FAISS index (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index(file_bytes: bytes, filename: str):
    """
    Builds the FAISS vector index from a PDF.
    Cached by Streamlit — only runs once per unique file.
    """
    # Save bytes to temp file (PyPDFLoader needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create embeddings (runs locally, no API needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def ask_claude(api_key: str, question: str, context_chunks: list, chat_history: list) -> str:
    """
    Calls Claude API with the retrieved context + conversation history.
    Much more reliable than HuggingFace Hub free tier.
    """
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in context_chunks
    ])

    # Build conversation history for Claude
    messages = []
    for exchange in chat_history[-4:]:  # last 4 exchanges to avoid hitting context limit
        messages.append({"role": "user", "content": exchange["question"]})
        messages.append({"role": "assistant", "content": exchange["answer"]})

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Answer the question below using ONLY the document excerpts provided.
If the answer is not in the excerpts, say "I couldn't find that in the document."
Always cite which page number(s) your answer comes from.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}"""
    })

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast + cheap, perfect for RAG
        max_tokens=1024,
        system="You are a precise document assistant. Answer questions using only the provided document excerpts. Always cite page numbers.",
        messages=messages
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# PROCESS BUTTON
# ─────────────────────────────────────────────
if uploaded_file and anthropic_key:
    if st.button("🚀 Process Document & Start Chatting"):
        if not anthropic_key.startswith("sk-ant"):
            st.error("That doesn't look like a valid Anthropic API key. It should start with sk-ant-")
        else:
            with st.spinner("Reading PDF → Chunking → Creating embeddings → Building index..."):
                try:
                    vs, num_chunks = build_index(uploaded_file.read(), uploaded_file.name)
                    st.session_state.vectorstore = vs
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✅ Ready! Indexed **{num_chunks} chunks** from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

elif uploaded_file and not anthropic_key:
    st.warning("⚠️ Enter your Anthropic API key above first.")

# ─────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────
if st.session_state.vectorstore:
    st.markdown(f"""
    <div class="status-ready">
        ✅ Ready: <strong>{st.session_state.doc_name}</strong> — Ask anything below
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 Ask Your Document")

    # Render chat history
    for exchange in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">🧑 {exchange["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">🔮 {exchange["answer"]}</div>', unsafe_allow_html=True)
        if exchange.get("sources"):
            pages = sorted(set([
                doc.metadata.get("page", 0) + 1
                for doc in exchange["sources"]
            ]))
            for p in pages:
                st.markdown(f'<span class="source-tag">📄 Page {p}</span>', unsafe_allow_html=True)

    # Input
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the main conclusion? / Summarize section 2 / What are the key findings?",
        key="q_input"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        ask_clicked = st.button("🔍 Ask PaperOracle")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_clicked and question:
        if not anthropic_key:
            st.error("Anthropic API key is missing. Please paste it above.")
        else:
            with st.spinner("Searching document and generating answer..."):
                try:
                    # Retrieve relevant chunks
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    source_docs = retriever.invoke(question)

                    # Get answer from Claude
                    answer = ask_claude(
                        api_key=anthropic_key,
                        question=question,
                        context_chunks=source_docs,
                        chat_history=st.session_state.chat_history
                    )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": source_docs
                    })
                    st.rerun()
                except anthropic.AuthenticationError:
                    st.error("Invalid Anthropic API key. Please check and try again.")
                except anthropic.RateLimitError:
                    st.error("Rate limit hit. Wait a moment and try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    # Empty state
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#484f58; padding: 40px 0;">
        <div style="font-size:3rem;">🔮</div>
        <div style="font-size:1rem; margin-top:10px;">Upload a PDF and enter your Anthropic API key to begin</div>
        <div style="font-size:0.84rem; margin-top:6px;">
            Get a free key at <strong>console.anthropic.com</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
