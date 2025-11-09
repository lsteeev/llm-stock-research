from langchain_core.prompts import ChatPromptTemplate

SAFETY_INSTRUCTION = """
CRITICAL INSTRUCTION: Only use information from the provided documents.
Do NOT:
- Fabricate data or statements
- Include generic boilerplate information
If information is insufficient to answer, clearly state:
"Insufficient information available in the provided documents."
"""

# Original single-pass prompt (kept for backward compatibility)
article_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are a senior equity research analyst writing a professional investment article in the style of Seeking Alpha.
{SAFETY_INSTRUCTION}

Your task:
- Produce a **cohesive narrative article** integrating qualitative and quantitative insights.
- Include discussion of the company's business model, products, financial performance, risks, and forward-looking statements.
- Integrate numbers naturally into the narrative; avoid bullet points, tables.
- Use flowing paragraphs with a professional, analytical tone.
- Limit to 3-5 paragraphs for readability.
- Only use facts explicitly stated in the provided context. If information is missing, clearly say so.

Context: {{context}}
"""),
    ("human", "Write a polished, investor-friendly narrative article based solely on the above context.")
])

# MAP Phase: Section-specific prompts
def create_section_prompt(section_title: str, section_focus: str) -> ChatPromptTemplate:
    """Create a prompt for generating a specific section of the report."""
    return ChatPromptTemplate.from_messages([
        ("system", f"""
You are a senior equity research analyst writing a specific section of an investment article in the style of Seeking Alpha.
{SAFETY_INSTRUCTION}

Your task:
- Write a focused section about: {section_focus}
- Use 2-3 well-structured paragraphs that integrate qualitative and quantitative insights.
- Integrate numbers naturally into the narrative; avoid bullet points and tables.
- Use a professional, analytical tone.
- Only use facts explicitly stated in the provided context. If information is missing, clearly state so.
- Focus specifically on {section_focus} - be detailed and specific.

Context: {{context}}
"""),
        ("human", f"Write a detailed section about {section_focus} for the investment article based solely on the above context.")
    ])

# Define report sections for MAP phase
REPORT_SECTIONS = [
    {
        "title": "Business Model & Operations",
        "focus": "the company's business model, core operations, products, services, and how the company generates revenue",
        "query_template": "business model operations products services revenue generation"
    },
    {
        "title": "Financial Performance",
        "focus": "financial performance including revenue, profitability, key financial metrics, trends, and comparisons across reporting periods",
        "query_template": "financial performance revenue profitability earnings cash flow financial metrics"
    },
    {
        "title": "Growth Strategy & Outlook",
        "focus": "growth strategies, expansion plans, market opportunities, forward-looking statements, and strategic initiatives",
        "query_template": "growth strategy expansion plans market opportunities forward looking strategic initiatives"
    },
    {
        "title": "Risks & Challenges",
        "focus": "risk factors, challenges, competitive threats, regulatory risks, and potential obstacles to the company's success",
        "query_template": "risk factors challenges competitive threats regulatory risks obstacles"
    },
    {
        "title": "Market Position & Competitive Landscape",
        "focus": "market position, competitive advantages, industry trends, and the company's standing relative to competitors",
        "query_template": "market position competitive advantages industry trends competitors"
    }
]

# REDUCE Phase: Combine sections into final article
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are a senior equity research analyst editing and refining an investment article in the style of Seeking Alpha.
{SAFETY_INSTRUCTION}

Your task:
- Combine the provided section drafts into a **cohesive, polished narrative article**.
- Ensure smooth transitions between sections.
- Eliminate redundancy and ensure consistency in tone and style.
- Integrate numbers naturally into the narrative; avoid bullet points and tables.
- Maintain a professional, analytical tone throughout.
- Create a flowing narrative that reads as a single, cohesive article (not separate sections).
- The final article should be comprehensive yet readable (aim for 5-8 paragraphs total).
- Only use information from the provided sections. Do not add new information not present in the sections.

Section drafts to combine:
{{sections}}

Company name: {{company_name}}
"""),
    ("human", "Combine the above sections into a polished, cohesive investment article. Ensure smooth transitions and eliminate redundancy.")
])