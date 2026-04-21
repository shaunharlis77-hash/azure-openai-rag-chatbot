# AI Document Intelligence Chatbot

A dual-mode Retrieval-Augmented Generation (RAG) system built with Streamlit, OpenAI, Azure Document Intelligence, Azure Blob Storage, and Azure AI Search.

This project demonstrates two complementary approaches to grounded document Q&A:

1. **Upload & Chat with Your Document**  
   A from-scratch RAG pipeline for uploaded PDFs using local chunking, embeddings, and retrieval.

2. **Azure Knowledge Base Q&A**  
   An enterprise-style RAG pipeline using Azure Blob Storage, Document Intelligence, and Azure AI Search for fast, pre-indexed document retrieval.

The system is designed to provide grounded answers, reduce hallucination, and demonstrate both core RAG fundamentals and cloud-based enterprise architecture.

## Features

- **Dual-mode RAG architecture**
  - Upload and chat with your own PDF
  - Query a pre-indexed Azure knowledge base

- **Grounded document Q&A**
  - Answers are generated from retrieved document chunks
  - Retrieved context is shown for transparency
  - Source files are displayed under answers

- **Upload mode**
  - PDF upload and save flow
  - Local document ingestion
  - Chunking and embedding generation
  - Similarity-based retrieval from scratch
  - Chat history and clear-history support

- **Azure knowledge base mode**
  - Azure Blob Storage as the document source
  - Azure Document Intelligence for extraction
  - Azure AI Search for vector retrieval
  - Fast query experience against pre-indexed documents
  - Chat history and clear-history support

- **Ingestion optimization**
  - Fast test mode using first-N-page processing
  - Full ingestion mode
  - Local caching to avoid unnecessary reprocessing of unchanged files

  ## Architecture

### Upload Mode (Local RAG Pipeline)

User uploads PDF  
→ Local extraction and chunking  
→ Embedding generation  
→ Similarity-based retrieval  
→ OpenAI generates grounded answer  

**Flow:**  
PDF → Chunking → Embeddings → Local Retrieval → OpenAI → Answer  

---

### Azure Knowledge Base Mode (Enterprise RAG Pipeline)

Documents stored in Azure Blob Storage  
→ Processed with Azure Document Intelligence  
→ Chunked and embedded  
→ Stored in Azure AI Search (vector index)  
→ Queried at runtime for fast retrieval  
→ OpenAI generates grounded answer  

**Flow:**  
Azure Blob Storage  
↓  
Document Intelligence  
↓  
Chunking + Embeddings  
↓  
Azure AI Search (Vector Index)  
↓  
OpenAI (Answer Generation)  
↓  
Streamlit UI  

---

### Ingestion Strategy

Azure ingestion is handled separately via:

`azure_ingest.py`

This script:
- Ensures the Azure AI Search index exists  
- Processes PDFs from Azure Blob Storage  
- Generates embeddings  
- Uploads chunked documents to Azure AI Search  

This design separates **batch ingestion** from **real-time querying**, improving performance and aligning with real-world enterprise systems.

## Tech Stack

- **Frontend / App**
  - Streamlit

- **LLM**
  - OpenAI (GPT-based models for answer generation)

- **Embeddings**
  - OpenAI Embeddings API

- **Local RAG Pipeline**
  - PyMuPDF (PDF parsing fallback)
  - Custom chunking + similarity search

- **Azure Services**
  - Azure Blob Storage (document storage)
  - Azure Document Intelligence (text extraction)
  - Azure AI Search (vector database and retrieval)

- **Other**
  - Python
  - JSON (for local caching)

  ## Setup Instructions

### 1. Clone the repository

    git clone <your-repo-url>
    cd <your-project-folder>

---

### 2. Create a virtual environment

**Windows:**

    python -m venv venv
    venv\Scripts\activate

**Mac/Linux:**

    python3 -m venv venv
    source venv/bin/activate

---

### 3. Install dependencies

    pip install -r requirements.txt

---

### 4. Configure environment variables

Create a `.env` file and add:

    OPENAI_API_KEY=your_key_here

    AZURE_SEARCH_ENDPOINT=your_endpoint
    AZURE_SEARCH_KEY=your_key
    AZURE_SEARCH_INDEX_NAME=rag-chunks

    AZURE_STORAGE_CONNECTION_STRING=your_connection_string

    DOCUMENT_INTELLIGENCE_ENDPOINT=your_endpoint
    DOCUMENT_INTELLIGENCE_KEY=your_key

---

### 5. Run Azure ingestion (one-time setup)

    python azure_ingest.py

---

### 6. Run the app

    streamlit run streamlit_app.py