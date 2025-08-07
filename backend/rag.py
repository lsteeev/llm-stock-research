from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import fitz
from typing import List, Dict, Any, Optional
import sys

PDF_PATH = "annual-report-agent/uploads/reports/Abbvie_2024.pdf"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
VECTORSTORE_DIR = "annual-report-agent/vectorstore/faiss"


def extract_headings(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract likely section headings from a PDF file using PyMuPDF (fitz).
    A heading is defined as a text span with <= 10 words and font size > 17.
    Returns a list of dicts with page number (1-based), text, and rounded font size.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}")

    headings: List[Dict[str, Any]] = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            lines = block.get("lines", [])
            for line in lines:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue  # Skip empty text
                    if len(text.split()) > 10 or span.get("size", 0) <= 17:
                        continue  # Not a heading by our heuristic
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
    """
    Assign the most recent heading (by page) to each chunk's metadata as 'section'.
    """
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        heading: Optional[str] = None
        for h in reversed(headings):
            if h["page"] <= page:
                heading = h["text"]
                break
        chunk.metadata["section"] = heading
    return chunks


def print_sample_chunks(chunks: List[Any], n: int = 5) -> None:
    print("\n--- Sample Chunks ---")
    for i, c in enumerate(chunks[:n]):
        print(f"\nChunk {i+1}")
        print("Content:", c.page_content[:200])
        print("Metadata:", c.metadata)


def print_retrieval_results(results: list) -> None:
    print("\n--- Retrieved Chunks with Scores ---")
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1}")
        print("Score:", score)
        print("Content:", doc.page_content[:200])
        print("Metadata:", doc.metadata)


def save_faiss_index(vectorstore, path):
    vectorstore.save_local(path)
    print(f"FAISS index saved to {path}")


def load_faiss_index(path, embedding):
    vectorstore = FAISS.load_local(path, embedding)
    print(f"FAISS index loaded from {path}")
    return vectorstore


def main(pdf_path: str = PDF_PATH, model_name: str = MODEL_NAME, mode: str = "rebuild") -> None:
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    if mode == "load":
        vectorstore = load_faiss_index(VECTORSTORE_DIR, embedding)
    else:
        print(f"Extracting headings from: {pdf_path}")
        headings = extract_headings(pdf_path)
        print(f"Found {len(headings)} headings.")
        print(headings)

        print("\nLoading PDF and splitting into chunks...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        cleaned_chunks = [clean_metadata(doc) for doc in chunks]
        cleaned_chunks = assign_sections_to_chunks(cleaned_chunks, headings)
        print_sample_chunks(cleaned_chunks)

        print("\nEmbedding and storing in FAISS...")
        vectorstore = FAISS.from_documents(cleaned_chunks, embedding)
        if mode == "save":
            save_faiss_index(vectorstore, VECTORSTORE_DIR)

    query = "New product"
    print(f"\nRetrieving top 3 chunks for query: '{query}'")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    print_retrieval_results(results_with_scores)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF chunking, embedding, and FAISS index management.")
    parser.add_argument("pdf_path", nargs="?", default=PDF_PATH, help="Path to PDF file")
    parser.add_argument("--save", action="store_true", help="Save FAISS index after building")
    parser.add_argument("--load", action="store_true", help="Load FAISS index from disk and skip embedding")
    args = parser.parse_args()

    if args.load:
        main(pdf_path=args.pdf_path, mode="load")
    elif args.save:
        main(pdf_path=args.pdf_path, mode="save")
    else:
        main(pdf_path=args.pdf_path, mode="rebuild")
