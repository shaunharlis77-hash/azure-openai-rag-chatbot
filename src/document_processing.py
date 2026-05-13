import fitz  # PyMuPDF
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

from src.config import DOCUMENT_INTELLIGENCE_ENDPOINT, DOCUMENT_INTELLIGENCE_KEY


def get_document_intelligence_client():
    if not DOCUMENT_INTELLIGENCE_ENDPOINT or not DOCUMENT_INTELLIGENCE_KEY:
        st.error("Document Intelligence endpoint or key is missing.")
        return None

    try:
        return DocumentIntelligenceClient(
            endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
        )
    except Exception as e:
        st.error(f"Error creating Document Intelligence client: {e}")
        return None


def extract_text_from_pdf(pdf_path, max_pages=None):
    text = ""

    try:
        doc = fitz.open(pdf_path)

        total_pages = len(doc)
        pages_to_read = total_pages if max_pages is None else min(max_pages, total_pages)

        for page_num in range(pages_to_read):
            page_text = doc[page_num].get_text("text")
            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")

    return text


def extract_text_from_pdf_bytes(pdf_bytes, max_pages=None):
    text = ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        total_pages = len(doc)
        pages_to_read = total_pages if max_pages is None else min(max_pages, total_pages)

        for page_num in range(pages_to_read):
            page_text = doc[page_num].get_text("text")
            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error reading PDF bytes: {e}")

    return text


def extract_text_with_document_intelligence(pdf_bytes, pages_param=None):
    client_di = get_document_intelligence_client()
    if client_di is None:
        return "", False

    try:
        if pages_param:
            poller = client_di.begin_analyze_document(
                model_id="prebuilt-read",
                body=pdf_bytes,
                pages=pages_param
            )
        else:
            poller = client_di.begin_analyze_document(
                model_id="prebuilt-read",
                body=pdf_bytes
            )

        result = poller.result()

        lines = []
        if result.pages:
            for page in result.pages:
                if page.lines:
                    for line in page.lines:
                        if line.content:
                            lines.append(line.content)

        extracted_text = "\n".join(lines)
        return extracted_text, True

    except Exception as e:
        st.warning(
            f"Document Intelligence extraction failed. Falling back to PyMuPDF. Error: {e}"
        )
        return "", False


def chunk_text(text, chunk_size=100, overlap=30):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        current_words = []
        current_length = 0
        i = start

        while i < len(words):
            word = words[i]
            additional_length = len(word) + (1 if current_words else 0)

            if current_length + additional_length <= chunk_size:
                current_words.append(word)
                current_length += additional_length
                i += 1
            else:
                break

        if current_words:
            chunks.append(" ".join(current_words))

        if i == len(words):
            break

        overlap_words = 0
        overlap_length = 0
        j = len(current_words) - 1

        while j >= 0 and overlap_length < overlap:
            overlap_length += len(current_words[j]) + 1
            overlap_words += 1
            j -= 1

        start += max(1, len(current_words) - overlap_words)

    return chunks