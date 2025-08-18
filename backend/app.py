from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import uvicorn
from typing import Dict, Any
import re

# --- App Setup ---
app = FastAPI()

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore/faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"  # Change this to your preferred Ollama model
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
RETRIEVAL_K = 3
SCORE_THRESHOLD = 0.2

# --- Load Components ---
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectorstore = FAISS.load_local(VECTORSTORE_PATH,
                              embeddings=embedding_model,
                              allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": SCORE_THRESHOLD, "k": RETRIEVAL_K}
)

# --- Ollama LLM Setup ---
llm = Ollama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
    num_predict=2048,  # Max tokens to generate
    timeout=60,  # Timeout in seconds
)

# --- Prompt Setup ---
system_prompt = """
You are a buy-side equity research analyst. 
Your role is to extract and analyze financial and strategic insights from company filings, reports, and commentary, 
and present them in a way that helps portfolio managers make investment decisions. 

Guidelines:
- Base your analysis strictly on the provided context. Do not use outside knowledge unless explicitly instructed. 
- Emphasize **financial performance, growth trends, margins, risks, and catalysts**. 
- Highlight **implications for investors** (e.g., earnings sustainability, competitive risks, pipeline strength). 
- Be concise but **analytical** — focus on what matters for an investment decision. 
- If multiple years of data are available, emphasize **directional trends and their significance**. 
- If information is missing, clearly state: 
  "The provided context does not contain this information."

Tone:
- Professional, objective, and investment-focused. 
- Avoid speculation, but highlight **risk/reward balance** when context allows. 
- Avoid casual or marketing-style language.

Format:
1. **Direct Answer** (1–2 sentences with key metric/trend).  
2. **Investment Insight**: explain why the data matters for investors (e.g., growth sustainability, margin pressure).  
3. **Supporting Details**: use bullet points with numbers and percentages for clarity.  
4. Always include units (%, millions, USD, etc.).

Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# --- Chain Composition ---
question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Request / Response Models ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    input: str
    context: Any
    think: str
    answer: str

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    try:
        # Test Ollama connection
        test_response = llm.invoke("Hello")
        return {"status": "healthy", "ollama": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        # Run the retrieval-augmented generation chain
        result = qa_chain.invoke({"input": req.query})

        # Raw answer from LLM (may contain <think> tags)
        raw_answer = result.get("answer", "")

        # Extract think content and remove it from final answer
        think_match = re.search(r"<think>([\s\S]*?)</think>", raw_answer)
        think_text = think_match.group(1).strip() if think_match else ""
        answer_no_think = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        return ChatResponse(
            input=result.get("input", req.query),
            context=result.get("context", {}),
            think=think_text,
            answer=answer_no_think,
        )

    except Exception as e:
        return ChatResponse(
            input=req.query,
            context="",
            think="",
            answer=f"Error processing request: {str(e)}",
        )

# --- Run Server ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print(f"Using Ollama model: {OLLAMA_MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print("Make sure Ollama is running with: ollama serve")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)