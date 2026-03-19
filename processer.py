from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.schema import retriever

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    print(f"Your OpenAI API Key is: {api_key}")

def load_text_file(file_path: str):
    loader = TextLoader(file_path)
    document = loader.load()
    return document

def chunk_text(document, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(document)
    return chunks

def embed_chunks(chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def create_vector_store(chunks, embedded_chunks):
    vector_store = FAISS.from_documents(chunks, embedded_chunks)
    vector_store.save_local('faiss_index')
    return vector_store

def query_vector_store(vector_store: FAISS, query: str):
    # results = vector_store.similarity_search(query, k=3)
    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"\nQuery: {query}")
    print("Results:\n")

    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        # print(f"Content: {doc.page_content}")
        # print(f"Metadata: {doc.metadata}\n")

        # print(doc[0])
        # print(doc[1])
        print(doc)
        print("-" * 50)



if __name__ == "__main__":
    main()

    document = load_text_file("data/sample1.txt")
    # print(document)
    # print(f"Document content: {document[0].page_content}")

    chunks = chunk_text(document)
    # print(chunks)
    # print(f"Number of chunks: {len(chunks)}")

    embedded_chunks = embed_chunks(chunks)
    # print(embedded_chunks)

    vector_store = create_vector_store(chunks, embedded_chunks)
    # print(vector_store)

    # retriever = vector_store.as_retriever()
    # print(retriever)

    query = "What is polar bears?"
    query_vector_store(vector_store, query)
