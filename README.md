# Hybrid Multi-Document RAG Chatbot with Real-Time Web Search

Link: https://multiragchatbot-c66q7jrsynrpq6xaxdtg8g.streamlit.app/

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that can answer questions using both **uploaded documents** and **real-time web search**.

The system combines **vector search using FAISS**, **semantic embeddings using HuggingFace**, **LLM inference using Groq**, and **real-time web search using Tavily** to generate grounded answers with source citations.

---

# Project Overview

Organizations store knowledge across many documents such as PDFs, reports, and notes. Traditional search systems struggle with understanding semantic meaning and combining information from multiple sources.

This project implements a **Hybrid RAG architecture** that allows users to:

- Upload documents
- Ask questions about those documents
- Retrieve relevant content using semantic search
- Augment answers with real-time web search
- Generate accurate answers using an LLM

The application is built using **Streamlit**, providing an interactive chat interface.

---

# Key Features

- Multi-document semantic search
- Retrieval-Augmented Generation (RAG)
- Real-time web search integration
- Document chunking and embedding
- FAISS vector database
- Streaming LLM responses
- Source citations for transparency
- Interactive Streamlit UI

---

# Tech Stack

| Component | Technology |
|--------|--------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.1 8B Instant) |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | FAISS |
| Web Search | Tavily Search API |
| Framework | LangChain |
| Language | Python |

---
## Screenshot

<img width="1864" height="960" alt="image" src="https://github.com/user-attachments/assets/8252788a-53b9-45d4-908c-73cad85b981c" />

---

# System Architecture

The system is organized into multiple layers:

User Interface Layer (Streamlit)

 ↓
 
Application Layer (app.py)

 ↓
 
Configuration Layer (settings.py)

↓

Core RAG Pipeline

 ↓
 
External Services (Groq, Tavily, HuggingFace)


---

# High-Level Architecture

The architecture consists of four major layers.

## 1. User Interface Layer

Handles interaction between the user and the system.

Components:

- Chat interface
- Document upload system
- Sidebar controls
- Web search toggle

Built using **Streamlit**.

---

## 2. Application Layer

This layer orchestrates the entire system.

Main responsibilities:

- Manage chat workflow
- Handle user queries
- Coordinate document processing
- Control the RAG pipeline


---

## 3. Core RAG Pipeline

This layer performs the retrieval-augmented generation process.

Modules include:

### Document Processor
Responsible for:

- Loading documents
- Cleaning text
- Splitting documents into chunks

### Embeddings Manager
Generates vector embeddings using:

```

sentence-transformers/all-MiniLM-L6-v2

```

### Vector Store Manager
Handles FAISS operations:

- Create vector index
- Store embeddings
- Perform similarity search

### RAG Chain
Handles:

- Query processing
- Context construction
- Prompt generation
- LLM response generation

---

## 4. External Services Layer

This layer integrates third-party AI services.

### Groq LLM

Used for response generation.

Model used:

```

llama-3.1-8b-instant

```

### Tavily Search

Provides real-time web search capabilities.

Used when:

- local documents lack relevant information
- user enables web search

### HuggingFace

Provides the embedding model used for semantic search.

---

# Low-Level Architecture

The system processes data through three major flows.

---

# 1. Document Ingestion Flow

When a user uploads documents, the following pipeline executes:

```

User Upload Documents
↓
Document Processor (Load & Parse)
↓
Text Splitter (Chunking)
↓
Embedding Generation (HuggingFace)
↓
Store Embeddings in FAISS Index

```

Steps explained:

1. User uploads `.pdf` or `.txt` documents.
2. Documents are parsed using LangChain loaders.
3. Text is split into smaller chunks.
4. Each chunk is converted into a vector embedding.
5. Embeddings are stored in a FAISS vector index.

---

# 2. Query Processing Flow

When the user asks a question:

```

User Query
↓
Query Embedding
↓
FAISS Similarity Search
↓
Retrieve Top-K Relevant Chunks
↓
Build Prompt (Context + Question)
↓
LLM Response Generation
↓
Stream Response to User

```

Steps explained:

1. The user query is converted into an embedding.
2. FAISS retrieves the most similar document chunks.
3. Retrieved chunks are assembled into context.
4. Context and query are injected into the prompt.
5. The LLM generates a grounded response.

---

# 3. Web Search Augmentation Flow

If document retrieval does not provide enough information, the system can use web search.

```

User Query
↓
Check if Local Context is Sufficient
↓
If Yes → Use Document Context
↓
If No → Perform Tavily Web Search
↓
Combine Results with Context
↓
Send to LLM
↓
Generate Final Response

```

This enables the chatbot to answer:

- recent events
- current statistics
- latest research updates

---

# Project Structure

```

project-root
│
├── app.py
│
├── config
│   └── settings.py
│
├── core
│   ├── chain.py
│   ├── document_processor.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── tools
│   └── tavily_search.py
│
├── ui
│   ├── chat_interface.py
│   └── components.py
│
├── data
│
├── requirements.txt
└── README.md


```

# Possible enhancements include:

Adding support for more document types

Implementing query classification

Improving hybrid retrieval strategies

Adding conversation memory



