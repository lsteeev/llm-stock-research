import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Config ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "deepseek-r1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
VECTORSTORE_DIR = "data/vectorstores"  # Match the directory from your RAG code

# Retrieval settings
RETRIEVAL_K = 6

# LLM generation settings
TEMPERATURE = 0.7
MAX_TOKENS = 2048
TIMEOUT = 180

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
    """
    Load FAISS vectorstore and return a retriever for the given company.
    
    Args:
        company_name: Company name (e.g., 'Victoria_Secret_2024')
    
    Returns:
        LangChain retriever object
    """
    vectorstore_path = os.path.join(VECTORSTORE_DIR, company_name)
    
    if not os.path.exists(vectorstore_path):
        raise ValueError(
            f"No vectorstore found for company '{company_name}'\n"
            f"Expected path: {os.path.abspath(vectorstore_path)}\n"
            f"Please build the index first using: python rag.py {company_name} --save"
        )
    
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )


def load_company_vectorstore(company_name: str):
    """
    Load FAISS vectorstore directly for the given company.
    Useful for custom queries without using the retriever interface.
    
    Args:
        company_name: Company name (e.g., 'Victoria_Secret_2024')
    
    Returns:
        FAISS vectorstore object
    """
    vectorstore_path = os.path.join(VECTORSTORE_DIR, company_name)
    
    if not os.path.exists(vectorstore_path):
        raise ValueError(
            f"No vectorstore found for company '{company_name}'\n"
            f"Expected path: {os.path.abspath(vectorstore_path)}\n"
            f"Please build the index first using: python rag.py {company_name} --save"
        )
    
    return FAISS.load_local(
        vectorstore_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


# --- Prompt Template ---
QA_PROMPT_TEMPLATE = """You are a financial analyst assistant. Use the following pieces of context from a company's annual report to answer the question at the end.

If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# --- Query Functions ---
def create_qa_chain(company_name: str):
    """
    Create a RetrievalQA chain for the given company.
    
    Args:
        company_name: Company name (e.g., 'Victoria_Secret_2024')
    
    Returns:
        RetrievalQA chain object
    """
    retriever = load_company_retriever(company_name)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    return qa_chain


def query_company(company_name: str, question: str, verbose: bool = True):
    """
    Query a company's annual report using RAG.
    
    Args:
        company_name: Company name (e.g., 'Victoria_Secret_2024')
        question: Question to ask about the company
        verbose: Whether to print source documents
    
    Returns:
        dict with 'result' and 'source_documents'
    """
    qa_chain = create_qa_chain(company_name)
    response = qa_chain.invoke({"query": question})
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Company: {company_name}")
        print(f"Question: {question}")
        print('='*60)
        print(f"\nAnswer:\n{response['result']}")
        print(f"\n{'='*60}")
        print(f"Sources ({len(response['source_documents'])} documents):")
        print('='*60)
        
        for i, doc in enumerate(response['source_documents'], start=1):
            print(f"\n[Source {i}]")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Section: {doc.metadata.get('section', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")
    
    return response


def manual_retrieval_query(company_name: str, question: str, k: int = 5):
    """
    Perform manual retrieval and display results without LLM generation.
    Useful for debugging and understanding what context is being retrieved.
    
    Args:
        company_name: Company name (e.g., 'Victoria_Secret_2024')
        question: Question/query
        k: Number of results to return
    """
    vectorstore = load_company_vectorstore(company_name)
    results = vectorstore.similarity_search_with_score(question, k=k)
    
    print(f"\n{'='*60}")
    print(f"Company: {company_name}")
    print(f"Query: {question}")
    print(f"Retrieved {len(results)} documents")
    print('='*60)
    
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n[Result {i}] Similarity Score: {score:.4f}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Section: {doc.metadata.get('section', 'N/A')}")
        print(f"Content: {doc.page_content[:300]}...")
        print('-' * 60)
    
    return results


# --- Main / Examples ---
def main():
    """Example usage of the query system."""
    company = "Victoria_Secret_2024"
    
    # Example queries
    questions = [
        "What are the main risk factors for this company?",
        "What was the revenue breakdown by segment?",
        "What are the company's growth strategies?",
    ]
    
    print(f"\n🔍 Querying {company} annual report...\n")
    
    for question in questions:
        try:
            response = query_company(company, question, verbose=True)
            print("\n" + "="*60 + "\n")
        except ValueError as e:
            print(f"❌ Error: {e}")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query company annual reports using RAG",
        epilog="""
Examples:
  # Interactive query
  python llm.py Victoria_Secret_2024 "What are the risk factors?"
  
  # Test retrieval only (no LLM)
  python llm.py Victoria_Secret_2024 --test-retrieval "revenue"
  
  # Run example queries
  python llm.py Victoria_Secret_2024 --examples
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "company_name",
        nargs="?",
        help="Company name (e.g., 'Victoria_Secret_2024')"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask about the company"
    )
    parser.add_argument(
        "--test-retrieval",
        metavar="QUERY",
        help="Test retrieval without LLM generation"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run example queries"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.examples or (not args.company_name):
            main()
        elif args.test_retrieval:
            if not args.company_name:
                print("❌ Error: Company name required for --test-retrieval")
                exit(1)
            manual_retrieval_query(args.company_name, args.test_retrieval, k=args.k)
        elif args.company_name and args.question:
            query_company(args.company_name, args.question, verbose=True)
        else:
            parser.print_help()
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise