import os
import json
import fitz  # PyMuPDF

PDF_FOLDER = "pdfs"
DATA_FOLDER = "data"
OUTPUT_FILE = os.path.join(DATA_FOLDER, "chunks.json")


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text


def chunk_text(text, chunk_size=100):
    chunks = []
    current_chunk = ""

    words = text.split()

    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def save_chunks_to_json(all_chunks, output_file):
    os.makedirs(DATA_FOLDER, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=4, ensure_ascii=False)


def load_chunks_from_json(output_file):
    if not os.path.exists(output_file):
        return []

    with open(output_file, "r", encoding="utf-8") as file:
        return json.load(file)


def keyword_search(query, chunks):
    query_words = query.lower().split()
    scored_chunks = []

    for chunk in chunks:
        content = chunk["content"].lower()
        score = sum(1 for word in query_words if word in content)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks


def generate_answer(query, results):
    if not results:
        return "I could not find an answer in the document."

    best_score, best_chunk = results[0]
    source = best_chunk["source_file"]
    chunk_number = best_chunk["chunk_number"]
    content = best_chunk["content"]

    answer = (
        f"Based on {source}, chunk {chunk_number}, "
        f"the answer to your question '{query}' is:\n\n"
        f"{content}"
    )

    return answer


def process_pdfs():
    if not os.path.exists(PDF_FOLDER):
        print(f"Folder not found: {PDF_FOLDER}")
        return

    pdf_files = [file for file in os.listdir(PDF_FOLDER) if file.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the pdfs folder.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s):\n")

    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Reading: {pdf_file}")

        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text.strip():
            chunks = chunk_text(extracted_text, chunk_size=100)
            print(f"Created {len(chunks)} chunk(s) from {pdf_file}")

            file_base = os.path.splitext(pdf_file)[0]

            for i, chunk in enumerate(chunks, start=1):
                chunk_record = {
                    "id": f"{file_base}_{i}",
                    "source_file": pdf_file,
                    "chunk_number": i,
                    "content": chunk
                }
                all_chunks.append(chunk_record)

        else:
            print(f"No text could be extracted from {pdf_file}.\n")

    save_chunks_to_json(all_chunks, OUTPUT_FILE)
    print(f"\nSaved {len(all_chunks)} total chunk(s) to: {OUTPUT_FILE}")


def search_chunks():
    if not os.path.exists(OUTPUT_FILE):
        print("No data found. Please run ingestion first (Option 1).")
        return

    chunks = load_chunks_from_json(OUTPUT_FILE)

    if not chunks:
        print("No chunks available to search.")
        return

    query = input("\nAsk a question about the document: ").strip()

    if not query:
        print("No question entered.")
        return

    results = keyword_search(query, chunks)

    if results:
        print("\nTop matching chunk(s):\n")

        for score, chunk in results[:3]:
            print(f"Score: {score}")
            print(f"Source File: {chunk['source_file']}")
            print(f"Chunk Number: {chunk['chunk_number']}")
            print(f"Content: {chunk['content']}")
            print("-" * 50)

        answer = generate_answer(query, results)
        print("\nGenerated Answer:\n")
        print(answer)

    else:
        print("No matching chunks found.")
        print("\nGenerated Answer:\n")
        print("I could not find an answer in the document.")


def main():
    while True:
        print("\n=== RAG Chatbot ===")
        print("1. Ingest PDFs")
        print("2. Ask Questions")
        print("3. Exit")

        choice = input("\nSelect an option (1/2/3): ").strip()

        if choice == "1":
            process_pdfs()
        elif choice == "2":
            search_chunks()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option selected.")


if __name__ == "__main__":
    main()