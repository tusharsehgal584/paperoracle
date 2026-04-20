# 🔮 PaperOracle — AI Document Q&A

> Upload any PDF. Ask questions in plain English. Get sourced answers powered by Claude AI.

Built with **LangChain · FAISS · HuggingFace Embeddings · Claude AI (Anthropic) · Streamlit**

---

## 🚀 Live Demo
[👉 Try PaperOracle on Streamlit Cloud](#) *(add your deployed link here)*

---

## 🧩 What It Does

- Upload **any PDF** (research paper, textbook, resume, report, contract)
- Ask **natural language questions** about the document
- Get **accurate, sourced answers** with page references
- Supports **follow-up questions** via conversation memory
- Powered by **Claude AI** — reliable, fast, no rate-limit headaches

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
Claude AI (claude-haiku-4-5)  →  generates grounded answer with page citations
    ↓
Answer + Source Pages displayed to user
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG Framework | LangChain |
| Embeddings | HuggingFace sentence-transformers (MiniLM) — runs locally |
| Vector Store | FAISS (Meta) |
| LLM | Claude Haiku via Anthropic API |
| PDF Loading | PyPDFLoader |

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

You'll need a free Anthropic API key from [console.anthropic.com](https://console.anthropic.com)

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

**Step 3 — (Optional) Pre-fill API key as a secret**
1. Streamlit Cloud dashboard → your app → **Settings → Secrets**
2. Add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
3. In `app.py`, replace the text_input line with:
   ```python
   anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
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

## 💡 Why Claude AI Instead of HuggingFace Hub LLM?

The original project used `HuggingFaceHub` LLM (Mistral-7B) which caused:
- Constant `401 Unauthorized` errors on free tier
- Slow response times (30–60 sec per answer)
- Deployment OOM crashes from loading `torch` + `transformers`

Claude Haiku via Anthropic API solves all three:
- Reliable auth with a simple API key
- Responses in ~1 second
- No heavy model loading — embeddings still run locally via sentence-transformers

---

## 👤 Author

**Tushar Sehgal**
- GitHub: [tusharsehgal584](https://github.com/tusharsehgal584)
- LinkedIn: [tusharsehgal-ai](https://www.linkedin.com/in/tusharsehgal-ai)
