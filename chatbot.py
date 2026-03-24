from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import os
from typing import Tuple, Any, Optional
from dotenv import load_dotenv

from processer import categorize_query

# Load environment variables
load_dotenv()
INDEX_NAME: str = os.getenv("INDEX_NAME", "faiss_index")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama3.1")

print(f"INDEX_NAME: {INDEX_NAME}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"AGENT_MODEL: {AGENT_MODEL}")

# Initialize embeddings and LLM
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ollama.Ollama(model=AGENT_MODEL)

# Lazy load summarization
_summarizer = None

def get_summarizer():
    """Lazy load summarizer to avoid slow startup time."""
    global _summarizer
    if _summarizer is None:
        print("Loading summarization model... (this may take a moment)")
        from summarization import summarize_text
        _summarizer = summarize_text
    return _summarizer


def load_vector_store(index_name: str = INDEX_NAME, query: str = "") -> Tuple[FAISS, Any]:
    """
    Load FAISS vector store, filter documents based on query categories.

    Args:
        index_name: Name of the FAISS index
        query: Query text to categorize and filter

    Returns:
        Tuple of (vector_store, retriever)
    """
    # Categorize query to filter context
    categories = categorize_query(query)
    print(f"Categories: {', '.join(categories)}")

    # Load vector store
    vector_store = FAISS.load_local(index_name, embedding, allow_dangerous_deserialization=True)

    # Search with similarity scores
    search_results = vector_store.similarity_search_with_score(", ".join(categories), k=10)

    # Filter results based on score threshold
    filtered_docs = [doc for doc, score in search_results if score / 1000 < 0.6]
    print(f"Filtered: {len(filtered_docs)} of {len(search_results)} documents")

    # Create filtered vector store
    docs_to_use = filtered_docs if filtered_docs else [search_results[0][0]]
    filtered_vector_store = FAISS.from_documents(docs_to_use, embedding)
    retriever = filtered_vector_store.as_retriever(search_kwargs={"k": 7})

    return (vector_store, retriever)


def summarize_response(context: Optional[str], question: str, response: str) -> str:
    """
    Summarize the response with context.

    Args:
        context: Previous context (or None for first query)
        question: User's question
        response: LLM response

    Returns:
        Summarized context
    """
    summarize_text = get_summarizer()

    if context:
        full_text = f"Context: {context}\nQuestion: {question}\nResponse: {response}"
    else:
        full_text = f"Question: {question}\nResponse: {response}"

    summarized = summarize_text(full_text)
    print(f"Summarized context: {summarized}")
    return summarized


def query_rag(question: str, context: Optional[str] = None) -> Tuple[str, str]:
    """
    Query the RAG system and return response and updated context.

    Args:
        question: User's question
        context: Previous context (or None for first query)

    Returns:
        Tuple of (response, updated_context)
    """
    loaded_vector_store, retriever = load_vector_store(INDEX_NAME, question)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type='stuff')

    # Build query with context if available
    if context:
        final_query = f"Context: {context}\nQuestion: {question}"
    else:
        final_query = question

    response = qa.run(final_query)
    updated_context = summarize_response(context, question, response)

    return response, updated_context


def main():
    """Main chatbot loop."""
    print("Starting the chatbot")
    context = ""

    while True:
        print("-" * 50)
        user_input = input("Enter your query (or 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        if not user_input:
            print("Please enter a valid query")
            continue

        try:
            response, context = query_rag(user_input, context if context else None)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    main()

