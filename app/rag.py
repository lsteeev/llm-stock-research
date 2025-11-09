from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz
import os
from typing import List, Dict, Any, Optional

PDF_DIR = "data/annual_reports"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 360
VECTORSTORE_DIR = "data/vectorstores"


# ---------- Utilities ----------
def resolve_pdf_path(name_or_path: str) -> str:
    """
    Convert company name to PDF path if needed.
    Examples:
      'Victoria_Secret_2024' -> 'data/annual_reports/Victoria_Secret_2024.pdf'
      'data/annual_reports/Victoria_Secret_2024.pdf' -> 'data/annual_reports/Victoria_Secret_2024.pdf'
    """
    # If it's already a valid file path, return it
    if os.path.isfile(name_or_path):
        return name_or_path
    
    # If it has .pdf extension but doesn't exist, raise error
    if name_or_path.endswith('.pdf'):
        if not os.path.isfile(name_or_path):
            raise FileNotFoundError(f"PDF file not found: {name_or_path}")
        return name_or_path
    
    # Otherwise, treat it as a company name and construct the path
    pdf_path = os.path.join(PDF_DIR, f"{name_or_path}.pdf")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n"
            f"Expected file at: {os.path.abspath(pdf_path)}"
        )
    return pdf_path


def extract_company_name(name_or_path: str) -> str:
    """
    Extract company name from either a file path or company name.
    Examples:
      'Victoria_Secret_2024' -> 'Victoria_Secret_2024'
      'data/annual_reports/Victoria_Secret_2024.pdf' -> 'Victoria_Secret_2024'
    """
    base_name = os.path.basename(name_or_path)
    return os.path.splitext(base_name)[0]


def extract_headings(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract likely section headings from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}")

    headings: List[Dict[str, Any]] = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    # Skip if too long or too short
                    if len(text.split()) > 10 or len(text) < 3:
                        continue
                    # Skip numbers, symbols, single characters
                    if text.startswith(('$', '>', '<', '%')) or text.isdigit():
                        continue
                    # Skip if mostly non-alphabetic
                    if sum(c.isalpha() for c in text) / len(text) < 0.5:
                        continue
                    if span.get("size", 0) <= 18:
                        continue
                    headings.append({
                        "page": page_num,
                        "text": text,
                        "size": round(span["size"], 2)
                    })
    return headings


def clean_metadata(doc) -> Any:
    """Keep only relevant metadata keys for a document chunk."""
    keep_keys = ["source", "page"]
    doc.metadata = {k: v for k, v in doc.metadata.items() if k in keep_keys}
    return doc


def assign_sections_to_chunks(chunks: List[Any], headings: List[Dict[str, Any]]) -> List[Any]:
    """Assign the most recent heading (by page) to each chunk."""
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        heading: Optional[str] = None
        for h in reversed(headings):
            if h["page"] <= page:
                heading = h["text"]
                break
        chunk.metadata["section"] = heading
    return chunks


def get_vectorstore_path(name_or_path: str) -> str:
    """Return consistent path for saving/loading FAISS index."""
    company_name = extract_company_name(name_or_path)
    return os.path.join(VECTORSTORE_DIR, company_name)


def save_faiss_index(vectorstore, name_or_path: str):
    save_path = get_vectorstore_path(name_or_path)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    company_name = extract_company_name(name_or_path)
    print(f"✅ FAISS index saved to {save_path}")
    print(f"💡 To load this index, use: --load {company_name}")


def load_faiss_index(name_or_path: str, embedding):
    load_path = get_vectorstore_path(name_or_path)
    if not os.path.exists(load_path):
        company_name = extract_company_name(name_or_path)
        raise FileNotFoundError(
            f"No FAISS index found at {load_path}\n"
            f"Please build the index first using: --save {company_name}"
        )
    vectorstore = FAISS.load_local(load_path, embedding, allow_dangerous_deserialization=True)
    company_name = extract_company_name(name_or_path)
    print(f"✅ FAISS index loaded from {load_path} for '{company_name}'")
    return vectorstore


# ---------- Core ----------
def build_vectorstore(name_or_path: str, model_name: str = MODEL_NAME, save: bool = True):
    """
    Extract chunks, embed, and build FAISS vectorstore from a PDF.
    Returns the vectorstore object.
    """
    pdf_path = resolve_pdf_path(name_or_path)
    company_name = extract_company_name(name_or_path)
    
    print(f"📄 Processing PDF: {pdf_path}")
    print(f"🏢 Company: {company_name}")

    # Headings
    headings = extract_headings(pdf_path)
    print(f"📑 Found {len(headings)} potential section headings")

    # Load + split PDF
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    cleaned_chunks = [clean_metadata(doc) for doc in chunks]
    cleaned_chunks = assign_sections_to_chunks(cleaned_chunks, headings)
    print(f"✂️  Created {len(cleaned_chunks)} chunks")

    # Embedding
    print(f"🔢 Creating embeddings with {model_name}...")
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(cleaned_chunks, embedding)

    # Save if needed
    if save:
        save_faiss_index(vectorstore, name_or_path)

    return vectorstore


# ---------- CLI / Debug ----------
def main(name_or_path: str, model_name: str = MODEL_NAME, mode: str = "rebuild"):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    company_name = extract_company_name(name_or_path)

    if mode == "load":
        vectorstore = load_faiss_index(name_or_path, embedding)
    else:
        vectorstore = build_vectorstore(name_or_path, model_name, save=(mode == "save"))

    # Quick test query
    test_queries = [
        "risk factors",
        "revenue by segment"
    ]

    print(f"\n{'='*60}")
    print(f"🔍 Testing RAG output for: {company_name}")
    print('='*60)

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        results = vectorstore.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n[Result {i}] Score: {score:.4f}")
            print(f"Page: {doc.metadata.get('page')}, Section: {doc.metadata.get('section')}")
            print(f"Content preview: {doc.page_content[:250]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="PDF chunking, embedding, and FAISS index management.",
        epilog="""
Examples:
  # Build and save index using company name
  python3 app/rag.py Victoria_Secret_2024 --save
  
  # Load existing index using company name
  python3 app/rag.py Victoria_Secret_2024 --load
  
  # Build without saving
  python3 app/rag.py Victoria_Secret_2024
  
  # Use full path
  python3 app/rag.py data/annual_reports/Victoria_Secret_2024.pdf --save
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "name_or_path", 
        nargs="?", 
        default="Victoria_Secret_2024",
        help="Company name (e.g., 'Victoria_Secret_2024') or full PDF path"
    )
    parser.add_argument("--save", action="store_true", help="Save FAISS index after building")
    parser.add_argument("--load", action="store_true", help="Load FAISS index from disk")
    args = parser.parse_args()

    try:
        if args.load:
            main(name_or_path=args.name_or_path, mode="load")
        elif args.save:
            main(name_or_path=args.name_or_path, mode="save")
        else:
            main(name_or_path=args.name_or_path, mode="rebuild")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise