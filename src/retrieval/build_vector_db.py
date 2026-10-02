import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.json"

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vectorstore"


# =========================
# SETTINGS
# =========================

COLLECTION_NAME = "crag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 32


# =========================
# LOAD CHUNKS
# =========================

def load_chunks():
    print("Loading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    return chunks


# =========================
# PREPARE DATA
# =========================

def prepare_chunks(chunks):
    texts = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        # Support the structure created by our chunker
        if isinstance(chunk, dict):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            )

            source = chunk.get("source", "unknown")

            page = chunk.get("page", 0)

        else:
            text = str(chunk)
            source = "unknown"
            page = 0

        text = text.strip()

        if not text:
            continue

        texts.append(text)

        metadatas.append({
            "source": str(source),
            "page": int(page) if str(page).isdigit() else 0
        })

        ids.append(f"chunk_{index}")

    return texts, metadatas, ids


# =========================
# BUILD VECTOR DATABASE
# =========================

def build_vector_database():

    chunks = load_chunks()

    texts, metadatas, ids = prepare_chunks(chunks)

    print(f"Valid chunks: {len(texts)}")

    print()
    print("Loading embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded")

    # Create directory
    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Creating ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    # Delete old collection if it exists
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Existing collection deleted")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    print("ChromaDB collection created")

    # =========================
    # EMBED + STORE
    # =========================

    total = len(texts)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = ids[start:end]

        print(
            f"Embedding chunks {start + 1}-{end} "
            f"of {total}"
        )

        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False
        )

        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadatas
        )

    print()
    print("================================")
    print("Vector database created!")
    print("================================")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total vectors: {collection.count()}")
    print(f"Saved to: {VECTOR_DB_DIR}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    build_vector_database()