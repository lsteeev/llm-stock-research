from pydantic import BaseModel
from typing import Any, List

# --- Chat Models ---
class ChatRequest(BaseModel):
    query: str
    company_name: str

class ChatResponse(BaseModel):
    input: str
    context: Any
    think: str
    answer: str

# --- Report Models ---
class ReportRequest(BaseModel):
    company_name: str = "Company"

class ReportSection(BaseModel):
    title: str
    content: str
    sources_used: List[str]

class ResearchReportResponse(BaseModel):
    company_name: str
    generated_at: str
    sections: List[ReportSection]
    full_report: str