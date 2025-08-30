from fastapi import FastAPI
from datetime import datetime
import uvicorn
import os

from llm import llm, load_company_retriever
from models import ChatRequest, ChatResponse, ReportRequest, ResearchReportResponse
from utils import extract_think_and_answer
from services.report import generate_report
from rag import build_vectorstore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompts import chat_prompt


app = FastAPI()

# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        # Load retriever for the requested company
        retriever = load_company_retriever(req.company_name)
        qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, chat_prompt))

        result = qa_chain.invoke({"input": req.query})
        raw_answer = result.get("answer", "")
        think_text, answer_no_think = extract_think_and_answer(raw_answer)

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

# --- Build Vectorstore Endpoint ---
@app.post("/build_vectorstore")
async def build_vectorstore_api(payload: dict):
    file_path = payload.get("file_path")
    file_name = payload.get("file_name")

    if not file_path or not os.path.exists(file_path):
        return {"status": "error", "message": "Invalid file path"}

    try:
        build_vectorstore(file_path, save=True)
        return {"status": "success", "message": f"Vectorstore built for {file_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Generate Report Endpoint ---
@app.post("/generate_report", response_model=ResearchReportResponse)
async def generate_report_api(req: ReportRequest) -> ResearchReportResponse:
    return await generate_report(req.company_name)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
