"""Add one PDF to the persistent corpus and vector index."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import uuid

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.pdf_loader import PDF_DIR
from src.retrieval.build_vector_db import BATCH_SIZE, COLLECTION_NAME, VECTOR_DB_DIR
from src.retrieval.retriever import load_embedding_model


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class UploadResult:
    filename: str
    pages: int
    chunks: int
    total_chunks: int


def safe_pdf_name(name: str) -> str:
    """Keep uploads inside the corpus directory with a readable source name."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    if Path(basename).suffix.lower() != ".pdf":
        raise ValueError("Upload a PDF file.")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._-")
    if not stem:
        raise ValueError("The PDF needs a valid filename.")
    return f"{stem}.pdf"


def pdf_chunks(content: bytes, source: str) -> tuple[list[Document], int]:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a nonempty PDF of at most 20 MB.")
    if not content.lstrip().startswith(b"%PDF-"):
        raise ValueError("The selected file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(content))
        pages = [Document(page_content=page.extract_text() or "",
                          metadata={"source": source, "page": number})
                 for number, page in enumerate(reader.pages, start=1)]
    except Exception as exc:
        raise ValueError("The PDF could not be read.") from exc
    readable = [page for page in pages if page.page_content.strip()]
    if not readable:
        raise ValueError("The PDF has no extractable text. Use a text-based PDF.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(readable), len(pages)


def add_pdf(name: str, content: bytes, *, collection=None, embedding_model=None,
            pdf_dir: Path = PDF_DIR) -> UploadResult:
    """Persist the PDF and index its chunks; roll back on indexing failure."""
    filename = safe_pdf_name(name)
    target = pdf_dir / filename
    if target.exists():
        raise ValueError(f"{filename} is already in the knowledge base. Rename the new file first.")
    chunks, page_count = pdf_chunks(content, filename)

    pdf_dir.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as saved:
        saved.write(content)

    upload_id = uuid.uuid4().hex
    identifiers = [f"upload_{upload_id}_{i}" for i in range(len(chunks))]
    try:
        if collection is None:
            VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            collection = chromadb.PersistentClient(path=str(VECTOR_DB_DIR)).get_or_create_collection(
                name=COLLECTION_NAME
            )
        if embedding_model is None:
            embedding_model = load_embedding_model()
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = [chunk.page_content for chunk in batch]
            vectors = embedding_model.encode(texts, show_progress_bar=False).tolist()
            collection.add(
                ids=identifiers[start:start + BATCH_SIZE],
                embeddings=vectors,
                documents=texts,
                metadatas=[chunk.metadata for chunk in batch],
            )
        return UploadResult(filename, page_count, len(chunks), collection.count())
    except Exception:
        if collection is not None:
            try:
                collection.delete(ids=identifiers)
            except Exception:
                pass
        target.unlink(missing_ok=True)
        raise
