"""Build the corpus and vector database using the team's ingestion modules."""
from src.ingestion.chunker import main as chunk_pdfs
from src.ingestion.pdf_loader import PDF_DIR


def main():
    if not any(PDF_DIR.glob("*.pdf")):
        raise SystemExit(f"No PDFs found. Put the team's corpus in {PDF_DIR}.")
    from src.retrieval.build_vector_db import build_vector_database

    chunk_pdfs()
    build_vector_database()


if __name__ == "__main__":
    main()
