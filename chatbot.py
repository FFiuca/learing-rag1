from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
from typing import Tuple, Union, Any
from dotenv import load_dotenv
from langchain_community.llms import ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
INDEX_NAME: str = os.getenv("INDEX_NAME", "faiss_index")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama3.1")

print(f"INDEX_NAME: {INDEX_NAME}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"AGENT_MODEL: {AGENT_MODEL}")

embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
def load_vector_store(index_name: str = INDEX_NAME) -> Tuple[FAISS, Any]:
    vector_store = FAISS.load_local(index_name, embedding, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    return (vector_store, retriever)

loaded_vector_store, retriever = load_vector_store(INDEX_NAME)
print(f"Vector store loaded: {loaded_vector_store}")

llm = ollama.Ollama(model=AGENT_MODEL)

# response = llm.invoke("What is polar bears?")
# print(f"LLM response: {response}")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type='stuff')
result = qa.run("WWho is Fikri Fiuca Fardana?")
print(f"QA Result: {result}")
