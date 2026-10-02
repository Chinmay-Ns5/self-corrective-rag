from pathlib import Path
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add ingestion folder to Python path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from pdf_loader import load_pdf


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = CURRENT_DIR.parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


# ---------------------------------------------------------
# CHUNKING SETTINGS
# ---------------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("Loading PDF documents...")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        print("ERROR: No PDF files found.")
        return

    documents = []

    # -----------------------------------------------------
    # LOAD ALL PDF PAGES
    # -----------------------------------------------------

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loaded_pages = load_pdf(pdf_file)

        for page in loaded_pages:

            # The loader currently returns dictionaries.
            # Convert them into LangChain Document objects.

            if isinstance(page, dict):

                # Try the common content field names.
                page_content = (
                    page.get("page_content")
                    or page.get("content")
                    or page.get("text")
                    or ""
                )

                page_metadata = page.get("metadata") or {}
                metadata = {
                    "source": page_metadata.get("source", pdf_file.name),
                    "page": page_metadata.get("page", 0),
                }

                document = Document(
                    page_content=page_content,
                    metadata=metadata
                )

            elif isinstance(page, Document):

                document = page

            else:

                print(f"WARNING: Unknown page type: {type(page)}")
                continue

            # Skip completely empty pages
            if document.page_content.strip():

                documents.append(document)

    print()
    print(f"Total pages loaded: {len(documents)}")


    # -----------------------------------------------------
    # CREATE TEXT SPLITTER
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


    # -----------------------------------------------------
    # SPLIT DOCUMENTS
    # -----------------------------------------------------

    print("Creating chunks...")

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")


    # -----------------------------------------------------
    # SAVE CHUNKS
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_data = []

    for chunk_id, chunk in enumerate(chunks):

        chunk_data.append({
            "chunk_id": chunk_id,
            "text": chunk.page_content,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
        })


    import json

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            chunk_data,
            f,
            indent=2,
            ensure_ascii=False
        )


    print()
    print("Chunking completed successfully!")
    print(f"Chunks saved to: {OUTPUT_FILE}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
