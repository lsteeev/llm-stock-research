import re
from typing import List

def extract_think_and_answer(raw_text: str) -> tuple[str, str]:
    """Extract think content and clean answer from LLM response."""
    think_match = re.search(r"<think>([\s\S]*?)</think>", raw_text)
    think_text = think_match.group(1).strip() if think_match else ""
    answer_no_think = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return think_text, answer_no_think

def extract_source_info(context_docs) -> List[str]:
    """Extract source information from context documents."""
    sources = []
    if context_docs:
        for doc in context_docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown source')
                if source not in sources:
                    sources.append(source)
    return sources[:3]  # Limit to 3 sources for brevity