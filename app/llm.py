import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# --- Config ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval settings
RETRIEVAL_K = 5
SCORE_THRESHOLD = 0.15

# LLM generation settings
TEMPERATURE = 0.7
MAX_TOKENS = 2048
TIMEOUT = 60


# --- Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# --- LLM (Ollama) ---
llm = Ollama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    timeout=TIMEOUT,
)


# --- Utility: Load company-specific retriever ---
def load_company_retriever(company_name: str):
    vectorstore_path = f"vectorstore/{company_name}"
    if not os.path.exists(vectorstore_path):
        raise ValueError(f"No vectorstore found for company '{company_name}'")

    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": SCORE_THRESHOLD, "k": RETRIEVAL_K}
    )