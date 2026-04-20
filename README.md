# 🔮 PaperOracle — AI Document Q&A

> Upload any PDF. Ask questions in plain English. Get sourced answers powered by Google Gemini AI.

Built with **LangChain · FAISS · HuggingFace Embeddings · Gemini 2.0 Flash · Streamlit**

---

## 🚀 Live Demo
[👉 Try PaperOracle](https://paperoracle.streamlit.app/) 

---

## 🧩 What It Does

- Upload **any PDF** — research paper, textbook, resume, report, contract
- Ask **natural language questions** about the document
- Get **accurate answers with page citations**
- Supports **follow-up questions** via conversation memory
- Powered by **Gemini 2.0 Flash** — completely free, no credit card needed

---

## 🏗️ How It Works (RAG Pipeline)

```
PDF Upload
    ↓
PyPDFLoader  →  extracts raw text page by page
    ↓
RecursiveCharacterTextSplitter  →  splits into 500-char overlapping chunks
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)  →  converts chunks to vectors (runs locally)
    ↓
FAISS Vector Store  →  indexes all vectors for similarity search
    ↓
User Question  →  embedded  →  top-3 similar chunks retrieved
    ↓
Gemini 2.0 Flash  →  generates grounded answer with page citations
    ↓
Answer + Source Pages displayed to user
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG Framework | LangChain |
| Embeddings | HuggingFace sentence-transformers (MiniLM) — runs locally, free |
| Vector Store | FAISS (Meta) |
| LLM | Gemini 2.0 Flash (Google AI) — free tier, 1500 req/day |
| PDF Loading | PyPDFLoader (pypdf) |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/tusharsehgal584/paperoracle
cd paperoracle

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Get a **free** Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — just sign in with Google, no credit card needed.

---

## 🚀 Deploy on Streamlit Cloud (Free)

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit — PaperOracle"
git branch -M main
git remote add origin https://github.com/tusharsehgal584/paperoracle.git
git push -u origin main
```

**Step 2 — Deploy**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your `paperoracle` repo → branch `main` → file `app.py`
4. Click **Deploy**

**Step 3 — (Optional) Pre-fill your API key as a secret**

In Streamlit Cloud → your app → **Settings → Secrets**, add:
```toml
GEMINI_API_KEY = "AIzaSy-your-key-here"
```
Then replace the `st.text_input` block in `app.py` with:
```python
gemini_key = st.secrets.get("GEMINI_API_KEY", "")
```

---

## 📁 Project Structure

```
paperoracle/
├── app.py              # Full Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 👤 Author

**Tushar Sehgal**
- GitHub: [tusharsehgal584](https://github.com/tusharsehgal584)
- LinkedIn: [tusharsehgal-ai](https://www.linkedin.com/in/tusharsehgal-ai)
