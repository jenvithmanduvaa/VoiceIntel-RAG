# AI Business Assistant (Agentic RAG)

Streamlit app that lets you ask business questions by voice or text. It loads the PDFs in `Code/data/`, chunks them, builds a FAISS vector store, and routes queries through a LangGraph agent: intent classify → optional decomposition → retrieval → answer with OpenAI models. Voice input is transcribed with Whisper; responses are shown in an expandable “behind the scenes” panel.

## Quick start
```bash
# from the project root (this folder)
python3 -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt

# add your OpenAI API key
export OPENAI_API_KEY="sk-..."

# run the Streamlit app
streamlit run Code/app.py
```

## App usage
- Open the URL Streamlit prints (usually http://localhost:8501).
- Speak or type a question; the app will retrieve from the PDF corpus and answer.
- Expand “Behind the Scenes” to see intents, sub-queries, doc names, and retrieved chunks.

## Data
Place PDFs in `Code/data/`. On startup they are loaded and embedded (FAISS). Current sample files:
- Company Profile & Strategic Overview.pdf
- Enterprise SaaS Competitive Analysis – AI Integration & Monetization Models 2025.pdf
- Q1/Q2 2025 SaaS/FMCG reports (4 files)

## Architecture (Code/agent.py)
- Embeddings: `OpenAIEmbeddings` over PDF chunks (`RecursiveCharacterTextSplitter`).
- Vector store: `langchain_community.vectorstores.FAISS`.
- Graph: `langgraph.StateGraph` with nodes classify → decompose → retrieve → answer.
- LLMs: `gpt-3.5-turbo` for fast steps, `gpt-4` for final answer (configurable).
- Caching: `InMemoryCache` for LangChain calls.

## Voice input
- Uses `streamlit-mic-recorder` for recording and `OpenAI` Whisper (`whisper-1`) for transcription. Requires audio input permissions.

## Notebook
Run the analysis notebook if needed:
```bash
cd "/Users/jenvithmanduva/GDS_Agentic RAG"
source .venv/bin/activate
jupyter notebook Code/Sales.ipynb
# or headless execution:
jupyter nbconvert --to notebook --execute Code/Sales.ipynb --output Code/Sales-ran.ipynb
```

## Notes
- If running offline initially, `tiktoken` may try to download tokenizer files; run once with internet to cache them.
- Keep your real `OPENAI_API_KEY` in an environment variable rather than hard-coding.***
