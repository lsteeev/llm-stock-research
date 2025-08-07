# Company Annual Report Research Agent

A web-based AI assistant for analyzing company annual reports, generating insights, and enabling natural language chat with the report.

## Features

- PDF upload & parsing
- Embedding & vector DB (FAISS/Chroma)
- Retrieval-Augmented Generation (RAG)
- Industry & market sentiment analysis
- Historical financial comparison
- Streamlit chat interface

## Folder Structure

```
annual-report-agent/
├── app.py                      # Main Streamlit app
├── backend/
│   ├── llm_interface.py        # Handles LLM calls
│   ├── embedding.py            # Embedding logic & vector DB ops
│   ├── pdf_parser.py           # PDF parsing logic
│   ├── retriever.py            # RAG logic for top-N chunks
│   ├── context_builder.py      # MCP implementation
│   ├── tools/
│   │   ├── news_sentiment.py   # Pull & summarize market sentiment
│   │   ├── finance_api.py      # Fetch historical KPI data
│   │   └── charts.py           # Generate trend charts
├── vectorstore/
│   └── db/                     # Vector DB index files
├── uploads/
│   └── reports/                # Uploaded PDFs
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repo and navigate to the project directory:**

   ```bash
   git clone <repo-url>
   cd annual-report-agent
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Configuration

- Add your API keys (OpenAI, NewsAPI, etc.) to a `.env` file in the root directory.

---

For more details, see the full project specification.
