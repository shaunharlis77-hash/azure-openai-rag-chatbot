# Azure OpenAI RAG Chatbot

> A dual-mode AI knowledge assistant built with Azure OpenAI, Azure AI Search, Azure Blob Storage, and Azure Document Intelligence.

![Hero Screenshot](screenshots/hero.png)

## Overview

Azure OpenAI RAG Chatbot is an enterprise-style Retrieval-Augmented Generation (RAG) platform designed to demonstrate real-world AI engineering patterns.

The system supports two distinct modes:

* **Local Document Workspace** — Upload and interact with PDFs using semantic search and grounded AI responses.
* **Enterprise Azure Knowledge Base** — Query a cloud-indexed Azure AI Search knowledge base powered by Azure services.

The project combines document ingestion, chunking, embeddings, semantic retrieval, and AI-generated responses inside a modular architecture with a polished product-style UI.

---

## Core Features

* Dual-mode RAG architecture
* Azure OpenAI-powered answer generation
* Azure AI Search vector retrieval
* Azure Document Intelligence extraction
* Azure Blob Storage integration
* Local PDF ingestion workflow
* Semantic chunk retrieval with embeddings
* Modular backend architecture
* Premium enterprise-style Streamlit UI
* Cached ingestion pipeline for faster reprocessing

---

## Architecture Snapshot

```text
User Query
    ↓
Embedding Generation
    ↓
Semantic Retrieval
    ↓
Context Injection
    ↓
Azure OpenAI Response Generation
```

### Azure Stack

* Azure OpenAI
* Azure AI Search
* Azure Blob Storage
* Azure Document Intelligence

---

## Preview

![Azure OpenAI RAG Chatbot](screenshots/app_preview6.png)

---

## Technical Highlights

* Modularized service architecture
* Retrieval-Augmented Generation pipeline
* Vector-based semantic search
* Enterprise-style UI redesign
* Local + cloud retrieval workflows
* Document Intelligence fallback handling
* Ingestion caching system

---

## Documentation

Detailed technical documentation and setup instructions:

- [Technical Overview](docs/technical_overview.md)

---

## Live Demo

[Access the live demo here.](https://azure-rag-chatbot.streamlit.app/)

---

## Author

Shaun Harlis

AI Engineer | Azure AI | Automation & Workflows | Real-World AI Solutions Builder
