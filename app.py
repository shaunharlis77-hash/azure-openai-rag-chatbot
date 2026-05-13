import os
from pathlib import Path

import streamlit as st

from src.answer_generation import generate_answer
from src.ingestion import (
    PDF_FOLDER,
    OUTPUT_FILE,
    load_chunks_from_json,
    process_pdfs,
)
from src.local_retrieval import keyword_search
from src.azure_search import search_azure_index
from ui.styles import load_custom_styles


def save_uploaded_files(uploaded_files):
    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(uploaded_file.name)

    return saved_files


st.set_page_config(page_title="Azure OpenAI Document RAG", layout="wide")
load_custom_styles()

# Azure knowledge base is indexed via azure_ingest.py

hero_col1, hero_col2 = st.columns([1.35, 1], gap="large")

with hero_col1:
    st.markdown('<div class="hero-badge">⚡ POWERED BY AZURE</div>', unsafe_allow_html=True)

    st.markdown(
    '<h1 class="hero-title">Azure OpenAI <span class="gradient-word">RAG Chatbot</span></h1>',
        unsafe_allow_html=True
    )

    st.markdown(
    '<p class="hero-copy">A dual-mode AI knowledge assistant that enables intelligent document interaction through local PDF analysis and enterprise-scale Azure AI Search.<br>Extract insights, retrieve grounded context, and generate accurate responses using Azure OpenAI, Document Intelligence, and Retrieval-Augmented Generation.</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="chip-row">'
        '<span class="chip">Azure OpenAI</span>'
        '<span class="chip">Azure AI Search</span>'
        '<span class="chip">Azure Blob Storage</span>'
        '<span class="chip">Azure Document Intelligence</span>'
        '</div>',
        unsafe_allow_html=True
    )

with hero_col2:
    hero_image_path = Path("assets") / "hero.png"

    if hero_image_path.exists():
        st.image(str(hero_image_path), use_container_width=True)
    else:
        st.warning(f"Hero image not found at: {hero_image_path}")

if "show_ingestion_settings" not in st.session_state:
    st.session_state.show_ingestion_settings = False

st.markdown('<div class="section-title">Choose Workspace</div>', unsafe_allow_html=True)

mode = st.radio(
    "",
    [
        "📄 Local Document Workspace",
        "☁️ Enterprise Azure Knowledge Base",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "📄 Local Document Workspace":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-heading">📄 Document Upload</div>
                <div class="card-subtitle">Upload PDFs and prepare them for grounded document chat.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Save Uploaded PDFs"):
                saved_files = save_uploaded_files(uploaded_files)
                st.session_state["recently_saved_files"] = saved_files
                st.success(f"Saved {len(saved_files)} file(s): {', '.join(saved_files)}")

        st.markdown("---")
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-heading">⚙️ Document Ingestion</div>
                <div class="card-subtitle">Extract content, create chunks, generate embeddings, and prepare the document for search.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Configure Ingestion"):
            st.session_state.show_ingestion_settings = True

        if st.session_state.show_ingestion_settings:
            with st.expander("Ingestion Settings", expanded=True):
                ingestion_mode = st.radio(
                    "Choose ingestion mode",
                    ["Fast test", "Full"],
                    horizontal=True,
                )

                fast_page_limit = None

                if ingestion_mode == "Fast test":
                    fast_page_limit = st.number_input(
                        "How many pages should be processed in Fast test mode?",
                        min_value=1,
                        max_value=50,
                        value=5,
                        step=1,
                    )

                if ingestion_mode == "Fast test":
                    pages_to_process = fast_page_limit
                    pages_param = f"1-{fast_page_limit}"
                else:
                    pages_to_process = None
                    pages_param = None

                st.markdown("#### Current configuration")
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
                            selected_files=selected_files,
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
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-heading">💬 Chat Interface</div>
                <div class="card-subtitle">Ask grounded questions about the PDF you uploaded and ingested.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []

        with st.form("search_form"):
            query = st.text_input("Enter your question", placeholder="Ask a question about your document...")
            submitted = st.form_submit_button("Search Document")

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
                        st.markdown("### Generated Answer")
                        with st.spinner("Thinking..."):
                            answer = generate_answer(query, results)

                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer,
                        })

                        st.success(answer)

                        source_files = list(set(chunk["source_file"] for _, chunk in results[:3]))

                        st.caption("Sources:")
                        for source in source_files:
                            st.write(f"• {source}")

                        st.markdown("### Retrieved Context")
                        for score, chunk in results[:3]:
                            with st.expander(
                                f"{chunk['source_file']} | Chunk {chunk['chunk_number']} | Score {score}"
                            ):
                                st.write(chunk["content"])
                    else:
                        st.warning("No matching chunks found.")
                        st.markdown("### Generated Answer")
                        st.write("I could not find an answer in the document.")

        if st.session_state.chat_history:
            st.markdown("### Chat History")

        for chat in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-user"><strong>You</strong><br>{chat["question"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="chat-bot"><strong>Assistant</strong></div>',
                unsafe_allow_html=True
            )

            st.markdown(chat["answer"])

elif mode == "☁️ Enterprise Azure Knowledge Base":
    if "azure_chat_history" not in st.session_state:
        st.session_state.azure_chat_history = []

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-heading">☁️ Enterprise Azure Knowledge Base</div>
            <div class="card-subtitle">
                Query a pre-indexed enterprise knowledge base built from Azure Blob Storage,
                Document Intelligence, and Azure AI Search.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear Azure Chat History"):
        st.session_state.azure_chat_history = []

    with st.form("azure_search_form"):
        user_question_azure = st.text_input(
            "Enter your question about the knowledge base:",
            placeholder="Ask a question about your Azure knowledge base...",
        )
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
                        "chunk_number": i,
                    }))

                st.markdown("### Generated Answer")
                with st.spinner("Generating answer..."):
                    azure_answer = generate_answer(user_question_azure, formatted_results)

                st.success(azure_answer)

                st.session_state.azure_chat_history.append({
                    "question": user_question_azure,
                    "answer": azure_answer,
                })

                source_files = list(set(chunk["source_file"] for chunk in azure_results))

                st.caption("Sources:")
                for source in source_files:
                    st.write(f"• {source}")

                st.markdown("### Retrieved Context")
                for i, chunk in enumerate(azure_results, start=1):
                    with st.expander(f"{chunk['source_file']} | Match {i}"):
                        st.write(chunk["content"])
            else:
                st.warning("No matching results found in Azure AI Search.")

    if st.session_state.azure_chat_history:
        st.markdown("### Chat History")

    for chat in st.session_state.azure_chat_history:
            st.markdown(
                f'<div class="chat-user"><strong>You</strong><br>{chat["question"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="chat-bot"><strong>Assistant</strong></div>',
                unsafe_allow_html=True
            )

            st.markdown(chat["answer"])