import os
import json
import re
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.search.documents.models import VectorizedQuery
from src.config import *
from src.document_processing import (
    get_document_intelligence_client,
    extract_text_from_pdf,
    extract_text_from_pdf_bytes,
    extract_text_with_document_intelligence,
    chunk_text,
)


PDF_FOLDER = "pdfs"
DATA_FOLDER = "data"
OUTPUT_FILE = os.path.join(DATA_FOLDER, "chunks.json")
INGESTION_CACHE_FILE = os.path.join(DATA_FOLDER, "ingestion_cache.json")

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-chunks")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)


def save_uploaded_files(uploaded_files):
    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(uploaded_file.name)

    return saved_files


def get_blob_service_client():
    if not AZURE_STORAGE_CONNECTION_STRING:
        st.error("Azure Storage connection string is missing.")
        return None

    try:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    except Exception as e:
        st.error(f"Error creating BlobServiceClient: {e}")
        return None


def list_blobs_in_container():
    blob_service_client = get_blob_service_client()
    if blob_service_client is None:
        return []

    try:
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs]
    except Exception as e:
        st.error(f"Error listing blobs: {e}")
        return []


def download_blob_to_bytes(blob_name):
    blob_service_client = get_blob_service_client()
    if blob_service_client is None:
        return None

    try:
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=blob_name
        )
        return blob_client.download_blob().readall()
    except Exception as e:
        st.error(f"Error downloading blob '{blob_name}': {e}")
        return None
    


    

def get_search_index_client():
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        st.error("Azure AI Search endpoint or key is missing.")
        return None

    try:
        return SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
    except Exception as e:
        st.error(f"Error creating SearchIndexClient: {e}")
        return None


def create_search_index():
    index_client = get_search_index_client()
    if index_client is None:
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),

        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="chunk_number",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),

        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True
        ),

        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="default-vector-profile"
        )
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="default-hnsw"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="default-hnsw"
            )
        ]
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    try:
        index_client.create_index(index)
        st.success(f"Search index '{AZURE_SEARCH_INDEX_NAME}' created successfully.")
    except Exception as e:
        st.warning(f"Index may already exist or failed to create: {e}")


def get_search_client():
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        st.error("Azure AI Search endpoint or key is missing.")
        return None

    try:
        return SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
    except Exception as e:
        st.error(f"Error creating SearchClient: {e}")
        return None        


def upload_chunks_to_search(chunk_documents):
    search_client = get_search_client()
    if search_client is None:
        return False

    try:
        results = search_client.upload_documents(documents=chunk_documents)
        successful_uploads = sum(1 for result in results if result.succeeded)

        st.success(f"Uploaded {successful_uploads} chunk(s) to Azure AI Search.")
        return True

    except Exception as e:
        st.error(f"Error uploading chunks to Azure AI Search: {e}")
        return False
    

def ensure_search_index_exists():
    index_client = get_search_index_client()
    if index_client is None:
        return False

    try:
        existing_index_names = [index.name for index in index_client.list_indexes()]

        if AZURE_SEARCH_INDEX_NAME not in existing_index_names:
            create_search_index()

        return True

    except Exception as e:
        st.error(f"Error checking or creating Azure AI Search index: {e}")
        return False


def search_azure_index(query, top_k=5):
    search_client = get_search_client()
    if search_client is None:
        return []

    try:
        query_embedding = get_embedding(query)

        if query_embedding is None:
            return []

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )

        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k
        )

        chunks = []
        for result in results:
            chunks.append({
                "content": result["content"],
                "source_file": result["source_file"]
            })

        return chunks

    except Exception as e:
        st.error(f"Error querying Azure AI Search: {e}")
        return []





def save_chunks_to_json(all_chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=4, ensure_ascii=False)
        

def load_chunks_from_json(output_file):
    if not os.path.exists(output_file):
        return []

    with open(output_file, "r", encoding="utf-8") as file:
        return json.load(file)


def load_ingestion_cache():
    if not os.path.exists(INGESTION_CACHE_FILE):
        return {}

    with open(INGESTION_CACHE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_ingestion_cache(cache_data):
    with open(INGESTION_CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache_data, file, indent=4)


def keyword_search(query, chunks):
    query_embedding = get_embedding(query)

    if query_embedding is None:
        return []

    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = chunk.get("embedding")

        if not chunk_embedding:
            continue

        score = cosine_similarity(query_embedding, chunk_embedding)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks

import re

def generate_answer(query, results):
    if not results:
        return "I could not find an answer in the document."

    top_chunks = []
    for _, chunk in results[:3]:
        top_chunks.append(chunk["content"])

    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a helpful AI assistant answering questions about uploaded documents.

Use the provided context as your primary source of truth.

Instructions:
- Answer clearly and naturally.
- Give a complete answer, not just a definition if more context is available.
- Keep it concise (2–3 sentences max).
- You may rephrase and lightly expand on the context to improve clarity.
- Do not introduce facts that are not supported by the context.
- Do not mention "the context" or "the document".
- If the answer cannot be reasonably inferred from the provided text, say exactly:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0.2
        )

        answer = response.output_text.strip()

        if not answer:
            return "I could not generate an answer."

        return f"{answer}\n\n_Source: {results[0][1]['source_file']}_"

    except Exception as e:
        return f"OpenAI error: {e}"
    

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None    


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    return dot_product / (norm_vec1 * norm_vec2)


def process_pdfs(pages_to_process=None, pages_param=None, selected_files=None):
    cache = load_ingestion_cache()

    if selected_files is not None:
        pdf_files = [file for file in selected_files if file.lower().endswith(".pdf")]
    else:
        pdf_files = [file for file in os.listdir(PDF_FOLDER) if file.lower().endswith(".pdf")]
    if not pdf_files:
        return 0, "No PDF files found in the pdfs folder.", [], [], []

    all_chunks = []
    di_used_files = []
    fallback_used_files = []
    di_failed_files = []
    cached_reused_files = []
    newly_processed_files = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        last_modified = os.path.getmtime(pdf_path)
        cached_file = cache.get(pdf_file)
        is_unchanged = (
            cached_file is not None and
            cached_file.get("last_modified") == last_modified and
            cached_file.get("pages_param") == pages_param
        )

        if is_unchanged:
            st.info(f"{pdf_file} is unchanged. Reusing cached chunks and skipping reprocessing.")
            cached_reused_files.append(pdf_file)
            cached_chunks = cached_file.get("chunks", [])
            all_chunks.extend(cached_chunks)
            continue
        else:
            st.info(f"{pdf_file} is new or changed. Processing.")
            newly_processed_files.append(pdf_file)

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception as e:
            st.error(f"Error reading local PDF '{pdf_file}': {e}")
            continue

        extracted_text, used_di = extract_text_with_document_intelligence(pdf_bytes, pages_param=pages_param)

        if used_di and extracted_text.strip():
            di_used_files.append(pdf_file)
        else:
            di_failed_files.append(pdf_file)
            extracted_text = extract_text_from_pdf(pdf_path, max_pages=pages_to_process)

            if extracted_text.strip():
                fallback_used_files.append(pdf_file)

        if extracted_text.strip():
            chunks = chunk_text(extracted_text, chunk_size=140, overlap=40)
            file_base = os.path.splitext(pdf_file)[0]
            file_chunks = []

            for i, chunk in enumerate(chunks, start=1):
                embedding = get_embedding(chunk)

                if embedding is None:
                    continue

                chunk_record = {
                    "id": f"{file_base}_{i}".replace(" ", "_"),
                    "source_file": pdf_file,
                    "chunk_number": i,
                    "content": chunk,
                    "embedding": embedding
                }
                all_chunks.append(chunk_record)
                file_chunks.append(chunk_record)

        cache[pdf_file] = {
            "last_modified": last_modified,
            "pages_param": pages_param,
            "chunks": file_chunks if extracted_text.strip() else []
        }

    save_chunks_to_json(all_chunks, OUTPUT_FILE)
    save_ingestion_cache(cache)
    return (
        len(all_chunks),
        "Local PDF ingestion completed.",
        di_used_files,
        fallback_used_files,
        di_failed_files,
        cached_reused_files,
        newly_processed_files
    )


def process_pdfs_from_azure(pages_to_process=None, pages_param=None):
    blob_names = list_blobs_in_container()

    if not blob_names:
        return 0, "No PDF blobs found in Azure Storage.", [], [], []

    all_chunks = []
    di_used_files = []
    fallback_used_files = []
    di_failed_files = []
    

    for blob_name in blob_names:
        if not blob_name.lower().endswith(".pdf"):
            continue

        pdf_bytes = download_blob_to_bytes(blob_name)

        if pdf_bytes is None:
            continue

        extracted_text, used_di = extract_text_with_document_intelligence(pdf_bytes, pages_param=pages_param)

        if used_di and extracted_text.strip():
            di_used_files.append(blob_name)
        else:
            di_failed_files.append(blob_name)
            extracted_text = extract_text_from_pdf_bytes(pdf_bytes, max_pages=pages_to_process)

            if extracted_text.strip():
                fallback_used_files.append(blob_name)

        if extracted_text.strip():
            chunks = chunk_text(extracted_text, chunk_size=140, overlap=40)
            file_base = os.path.splitext(blob_name)[0]

            for i, chunk in enumerate(chunks, start=1):
                embedding = get_embedding(chunk)

                if embedding is None:
                    continue

                chunk_record = {
                    "id": f"{file_base}_{i}".replace(" ", "_"),
                    "source_file": blob_name,
                    "chunk_number": i,
                    "content": chunk,
                    "embedding": embedding
                }
                all_chunks.append(chunk_record)

    save_chunks_to_json(all_chunks, OUTPUT_FILE)

    if ensure_search_index_exists():
        upload_chunks_to_search(all_chunks)

    return (
        len(all_chunks),
        "Azure PDF ingestion completed.",
        di_used_files,
        fallback_used_files,
        di_failed_files
)


st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Azure knowledge base is indexed via azure_ingest.py

st.title("🧠 AI Document Intelligence Chatbot")
st.caption("A dual-mode RAG system for uploaded PDFs and an Azure-powered enterprise knowledge base.")


if "show_ingestion_settings" not in st.session_state:
    st.session_state.show_ingestion_settings = False
    
st.subheader("Choose Mode")

tab1, tab2 = st.tabs([
    "📄 Upload & Chat with Your Document",
    "☁️ Azure Knowledge Base"
])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.divider()
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Save Uploaded PDFs"):
                saved_files = save_uploaded_files(uploaded_files)
                st.session_state["recently_saved_files"] = saved_files
                st.success(f"Saved {len(saved_files)} file(s): {', '.join(saved_files)}")

        st.divider()
        st.subheader("Ingest Your Document")
        st.write("Upload a PDF, save it, ingest it, and ask grounded questions about its contents.")
    
        st.write("Ready to ingest your uploaded PDF(s)?")

        if st.button("Ingest Uploaded PDFs"):
            st.session_state.show_ingestion_settings = True

        if st.session_state.show_ingestion_settings:
            with st.expander("Ingestion Settings", expanded=True):
                ingestion_mode = st.radio(
                    "Choose ingestion mode",
                    ["Fast test", "Full"],
                    horizontal=True
                )

            fast_page_limit = None

            if ingestion_mode == "Fast test":
                fast_page_limit = st.number_input(
                    "How many pages should be processed in Fast test mode?",
                    min_value=1,
                    max_value=50,
                    value=5,
                    step=1
                )

            if ingestion_mode == "Fast test":
                pages_to_process = fast_page_limit
                pages_param = f"1-{fast_page_limit}"
            else:
                pages_to_process = None
                pages_param = None

            st.write("### Current ingestion configuration")
            st.write(f"Mode: {ingestion_mode}")

            if pages_to_process:
                st.write(f"Pages to process: first {pages_to_process} pages")
            else:
                st.write("Pages to process: full document")

            if st.button("Run Ingestion"):
                selected_files = st.session_state.get("recently_saved_files", [])

                if not selected_files:
                    st.warning("No uploaded PDFs found in this session. Please upload and save a PDF first.")
                else:
                    chunk_count, message, di_used_files, fallback_used_files, di_failed_files, cached_reused_files, newly_processed_files = process_pdfs(
                        pages_to_process=pages_to_process,
                        pages_param=pages_param,
                        selected_files=selected_files
                    )

                    if chunk_count > 0:
                        st.success(f"{message} Total chunks created: {chunk_count}")
                    else:
                        st.warning(message)

                    if di_used_files:
                        st.info("Document Intelligence used for:")
                        for file_name in di_used_files:
                            st.write(f"✅ {file_name}")

                    if fallback_used_files:
                        st.warning("Fallback to PyMuPDF used for:")
                        for file_name in fallback_used_files:
                            st.write(f"⚠️ {file_name}")

                    if di_failed_files:
                        st.error("Document Intelligence failed or exceeded limits for:")
                        for file_name in di_failed_files:
                            st.write(f"❌ {file_name}")

                    if cached_reused_files:
                        st.info("Reused from cache:")
                        for file_name in cached_reused_files:
                            st.write(f"♻️ {file_name}")

                    if newly_processed_files:
                        st.info("Newly processed:")
                        for file_name in newly_processed_files:
                            st.write(f"🆕 {file_name}")

   
    with col2:
        st.divider()
        st.subheader("Chat with Your Document")
        st.write("Ask grounded questions about the PDF you uploaded and ingested.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []

        with st.form("search_form"):
            query = st.text_input("Enter your question")
            submitted = st.form_submit_button("Search")
    
        if submitted:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                chunks = load_chunks_from_json(OUTPUT_FILE)

                if not chunks:
                    st.warning("No chunk data found. Please ingest PDFs first.")
                else:
                    results = keyword_search(query, chunks)

                    if results:
                        st.subheader("Generated Answer")
                        with st.spinner("Thinking..."):
                            answer = generate_answer(query, results)

                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer
                        })

                        st.success(answer)

                        source_files = list(set(chunk["source_file"] for _, chunk in results[:3]))

                        st.caption("Sources:")
                        for source in source_files:
                            st.write(f"• {source}")
                    
                        st.subheader("Top Matching Chunks")
                        for score, chunk in results[:3]:
                            with st.expander(
                                f"{chunk['source_file']} | Chunk {chunk['chunk_number']} | Score {score}"
                            ):
                                st.write(chunk["content"])
                    else:
                        st.warning("No matching chunks found.")
                        st.subheader("Generated Answer")
                        st.write("I could not find an answer in the document.")

        if st.session_state.chat_history:
            st.markdown("### Chat History")

        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            st.markdown("---")                


with tab2:
    if "azure_chat_history" not in st.session_state:
        st.session_state.azure_chat_history = []

    st.divider()
    st.subheader("Chat with Azure Knowledge Base")
    st.write(
        "Query a pre-indexed enterprise knowledge base built from Azure Blob Storage using "
        "Document Intelligence and Azure AI Search for fast, grounded responses."
    )

    if st.button("Clear Azure Chat History"):
        st.session_state.azure_chat_history = []

    with st.form("azure_search_form"):
        user_question_azure = st.text_input("Enter your question about the knowledge base:")
        submitted = st.form_submit_button("Ask Azure Knowledge Base")

    if submitted:
        if not user_question_azure:
            st.warning("Please enter a question.")
        else:
            azure_results = search_azure_index(user_question_azure, top_k=5)

            if azure_results:
                formatted_results = []
                for i, chunk in enumerate(azure_results, start=1):
                    formatted_results.append((1.0, {
                        "content": chunk["content"],
                        "source_file": chunk["source_file"],
                        "chunk_number": i
                    }))

                st.subheader("Generated Answer")
                with st.spinner("Generating answer..."):
                    azure_answer = generate_answer(user_question_azure, formatted_results)

                st.success(azure_answer)

                st.session_state.azure_chat_history.append({
                    "question": user_question_azure,
                    "answer": azure_answer
                })

                # Extract unique source files
                source_files = list(set(chunk["source_file"] for chunk in azure_results))

                st.caption("Sources:")
                for source in source_files:
                    st.write(f"• {source}")

                st.subheader("🔍 Retrieved Context")
                for i, chunk in enumerate(azure_results, start=1):
                    with st.expander(f"{chunk['source_file']} | Match {i}"):
                        st.write(chunk["content"])
            else:
                st.warning("No matching results found in Azure AI Search.")

    if st.session_state.azure_chat_history:
        st.markdown("### Chat History")

    for chat in st.session_state.azure_chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        st.markdown("---")