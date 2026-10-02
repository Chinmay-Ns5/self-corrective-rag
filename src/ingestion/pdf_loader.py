from pathlib import Path
from pypdf import PdfReader


# Project root:
# self-corrective-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"


def load_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from a single PDF.

    Returns one dictionary per page containing:
    - extracted text
    - source filename
    - page number
    """

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "page": page_number,
                },
            }
        )

    return documents


def load_all_pdfs() -> list[dict]:
    """
    Load all PDF files from data/raw/pdfs/.
    """

    all_documents = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PDF_DIR}"
        )

    for pdf_file in pdf_files:
        documents = load_pdf(pdf_file)
        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":
    documents = load_all_pdfs()

    print(f"PDF directory: {PDF_DIR}")
    print(f"Total pages loaded: {len(documents)}")

    if documents:
        print("\nFirst document:")
        print(f"Source: {documents[0]['metadata']['source']}")
        print(f"Page: {documents[0]['metadata']['page']}")
        print(f"Characters: {len(documents[0]['text'])}")

        print("\nFirst 500 characters:")
        print(documents[0]["text"][:500])