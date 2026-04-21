from streamlit_app import ensure_search_index_exists, process_pdfs_from_azure

def main():
    ensure_search_index_exists()
    process_pdfs_from_azure()

if __name__ == "__main__":
    main()