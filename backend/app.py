from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import uvicorn
from typing import Dict

# --- App Setup ---
app = FastAPI()

# --- Configuration ---
VECTORSTORE_PATH = "annual-report-agent/vectorstore/faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "microsoft/phi-2"  # or any other small HF model
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

# --- HF Pipeline ---
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
hf_pipe = pipeline(
    "text-generation",
    model=LLM_MODEL_NAME,
    device=-1,  # CPU
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=hf_pipe)

# --- Prompt Setup ---
system_prompt = """
You are a helpful financial analyst. Use the context below to answer the question.

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
    response: str

# --- Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result: Dict[str, str] = qa_chain.invoke({"input": req.query})
    return ChatResponse(response=result["answer"])

# --- Run Server ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
