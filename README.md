# RAG with Filtering and Memorizing System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines **intelligent document filtering** through zero-shot classification and **context memorization** through summarization, enabling context-aware conversations with dynamic document retrieval.

## 🎯 Project Overview

This project implements a multi-layered RAG system that:

1. **Filters Documents** - Uses zero-shot classification to categorize queries and retrieve only relevant documents
2. **Retrieves Context** - Leverages FAISS vector stores with semantic similarity search
3. **Generates Responses** - Powers responses with a local LLM agent
4. **Memorizes Conversation** - Maintains conversation context through intelligent summarization

### Key Features

- ✅ **Category-based Filtering** - Automatically categorize queries and filter document retrieval
- ✅ **Semantic Search** - FAISS-based vector similarity search for relevant documents
- ✅ **Context Memorization** - Summarize conversation history to maintain long-term context
- ✅ **Lazy Loading** - Fast startup with on-demand model loading
- ✅ **MongoDB Integration** - Store categories and metadata for filtering
- ✅ **Modular Design** - Clean separation of concerns with reusable components

---

## 🧠 Models Used

### Document Processing & Filtering

| Component | Model | Purpose |
|-----------|-------|---------|
| **Zero-Shot Classifier** | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Categorize documents and queries without labeled training data |
| **Embeddings** | `nomic-embed-text` | Generate semantic embeddings for documents and queries |

### Response Generation & Memorization

| Component | Model | Purpose |****
|-----------|-------|---------|
| **Agent/LLM** | `llama3.1` (Ollama) | Generate responses based on retrieved context |
| **Summarizer** | `facebook/bart-large-cnn` | Summarize conversation history for context retention |

### Vector Store

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Fast similarity search over document embeddings |
| **Metadata Storage** | MongoDB | Store categories, documents, and filtering metadata |

---

## 🏗️ System Architecture

### Processer Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

                              START
                                │
                                ▼
                    ┌───────────────────┐
                    │  Load Text Files  │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────────────────────┐
                    │  Zero-Shot Classification         │
                    │  (Categorize Documents)           │
                    │  Model: mDeBERTa-v3-base-mnli-xnli│
                    └────────┬──────────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │  Save to MongoDB     │
                    │  - Text             │
                    │  - Categories       │
                    │  - Metadata         │
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │  Chunk Documents     │
                    │  (RecursiveCharacter)│
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │  Generate Embeddings │
                    │  Model: nomic-embed- │
                    │         text         │
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │  Create FAISS Index  │
                    │  (Vector Store)      │
                    └────────┬─────────────┘
                             │
                             ▼
                           SUCCESS
```

### Chatbot Query Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

                              START
                                │
                   ┌────────────┴────────────┐
                   │  User Query Input       │
                   │  + Previous Context     │
                   └────────────┬────────────┘
                                │
                                ▼
                    ┌─────────────────────────────────┐
                    │  Categorize Query               │
                    │  (Zero-Shot Classification)     │
                    │  Model: mDeBERTa-v3-base       │
                    └────────┬────────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │  Search FAISS Vector Store   │
                    │  - Join category labels      │
                    │  - Similarity search (k=10)  │
                    │  - Filter by score (< 0.6)   │
                    └────────┬─────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │  Create Retriever            │
                    │  (Top-7 filtered documents)  │
                    └────────┬─────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │  Build Query with Context    │
                    │  If context exists:          │
                    │  "Context: {context}         │
                    │   Question: {question}"      │
                    └────────┬─────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │  Generate Response           │
                    │  Model: llama3.1 (Ollama)    │
                    │  Using RetrievalQA Chain     │
                    └────────┬─────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │  Summarize Conversation      │
                    │  Model: facebook/bart-large  │
                    │  Input: Context+Q+Response   │
                    └────────┬─────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │  Return Response             │
                    │  + Updated Context           │
                    └────────┬─────────────────────┘
                             │
                             ▼
                           END
```

---

## 📁 Project Structure

```
provider-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── db.py                        # MongoDB connection module
├── processer.py                 # Document processing pipeline
├── chatbot.py                   # RAG query pipeline
├── summarization.py             # Summarization module
│
├── data/
│   ├── sample1.txt             # Wildlife content (Polar Bears)
│   └── sample2.txt             # Additional sample data
│
├── faiss_index/
│   ├── index.faiss             # FAISS vector index
│   └── index.pkl               # Index metadata
│
└── seeder/
    └── category.ipynb          # MongoDB category seeder notebook
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- MongoDB (running locally or remote)
- Ollama (for embedding and LLM models)

### 2. Installation

```bash
# Create virtual environment
python -m venv env-rag2
source env-rag2/Scripts/activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create `.env` file in `provider-project/`:

```env
# Models
EMBEDDING_MODEL="nomic-embed-text"
AGENT_MODEL="llama3.1"
SUMMARIZATION_MODEL="facebook/bart-large-cnn"
CLASSIFIER_MODEL="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# FAISS Index
INDEX_NAME="faiss_index"

# MongoDB
MONGO_URI="mongodb://localhost:27017"
MONGO_DB_NAME="rag_filtering"
```

### 4. Seed Categories (First Time Only)

```bash
# Navigate to seeder directory and run category.ipynb
# This will seed 10 categories to MongoDB:
# Wildlife, Geography, History, Science, Technology,
# Education, Health, Sports, Entertainment, Politics
```

### 5. Process Documents

```bash
# Add your .txt files to data/ directory
# Then run processer.py to:
# - Categorize documents
# - Save metadata to MongoDB
# - Create FAISS vector index

python processer.py
```

### 6. Start Chatbot

```bash
python chatbot.py
```

---

## 💡 How It Works

### Document Filtering

1. **Query Categorization**: User query is classified into 0+ categories using zero-shot classification
2. **Category Matching**: Only documents tagged with matching categories are retrieved
3. **Similarity Filtering**: Additional similarity scoring (FAISS) filters by relevance score < 0.6

### Context Memorization

1. **Query Embedding**: Question is embedded using `nomic-embed-text`
2. **Semantic Retrieval**: Top-10 similar documents retrieved from FAISS
3. **Context Building**: Previous context is prepended to the query
4. **Response Generation**: LLM generates response using retrieved context
5. **Summary Creation**: BART summarizer compresses conversation history into key points

---

## 📊 Data Flow

```
User Input
    ↓
Query Categorization (mDeBERTa-v3)
    ↓
FAISS Similarity Search
    ↓
Retrieved Documents (filtered)
    ↓
LLM Response Generation (llama3.1)
    ↓
Context Summarization (BART)
    ↓
Response + Updated Memory
```

---

## 🔧 Core Components

### `processer.py`
- **`categorize_query()`** - Zero-shot classification for document categorization
- **`load_text_file()`** - Load documents from disk
- **`chunk_text()`** - Split documents into semantic chunks
- **`create_vector_store()`** - Create and persist FAISS index
- **`process_documents()`** - End-to-end processing pipeline

### `chatbot.py`
- **`load_vector_store()`** - Load FAISS index with category filtering
- **`query_rag()`** - Main RAG pipeline: retrieve → generate → summarize
- **`summarize_response()`** - Update conversation memory
- **`main()`** - Interactive chatbot loop

### `db.py`
- **`get_mongo_db()`** - MongoDB connection (singleton pattern)
- **`get_mongo_collection()`** - Access specific collections
- **Utility functions** - Create, find, update, delete documents

### `summarization.py`
- **`summarize_text()`** - Summarize text using BART model
- Lazy-loaded for fast startup

---

## 🎓 Example Conversation

```
>>> Enter your query: Tell me about polar bears
Categories: ['Wildlife']
Filtered: 7 of 10 documents

Response: Polar bears are apex predators of the Arctic...

>>> Enter your query: How do they hunt?
Categories: ['Wildlife']
[Using previous context about polar bears...]

Response: Polar bears are patient hunters that wait at breathing holes...
Summarized context: Polar bears are Arctic apex predators that hunt seals
using patience and camouflage techniques...

>>> Enter your query: Are they endangered?
Categories: ['Wildlife']
[Using accumulated context...]

Response: Yes, climate change is the primary threat...
```

---

## 📋 Requirements

Key dependencies:

```
langchain-community >= 0.3.0
langchain-text-splitters >= 0.2.0
faiss-cpu >= 1.7.4
pymongo >= 4.6.0
transformers >= 4.36.0
torch >= 2.1.0
ollama >= 0.1.0
python-dotenv >= 1.0.0
```

See `requirements.txt` for complete list.

---

## 🔐 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `nomic-embed-text` | Document embedding model |
| `AGENT_MODEL` | `llama3.1` | LLM for response generation |
| `CLASSIFIER_MODEL` | `mDeBERTa-v3-base-mnli-xnli` | Zero-shot classifier |
| `SUMMARIZATION_MODEL` | `facebook/bart-large-cnn` | Context summarization |
| `INDEX_NAME` | `faiss_index` | FAISS index directory |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `MONGO_DB_NAME` | `rag_filtering` | Database name |

---

## 🤝 Contributing

To extend this project:

1. Add more category types in `category.ipynb` seeder
2. Experiment with different classifier or summarization models
3. Implement custom filtering logic in `load_vector_store()`
4. Add persistence for conversation history

---

## 📝 License

MIT License - Feel free to use for educational and research purposes.

---

## 🔗 References

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [MongoDB Python Driver](https://pymongo.readthedocs.io/)
- [Ollama](https://ollama.ai/)

---

**Last Updated**: March 29, 2026

**Status**: ✅ Production Ready
