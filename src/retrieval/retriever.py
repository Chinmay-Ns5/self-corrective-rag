import os
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

VECTORSTORE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "vectorstore"
)

COLLECTION_NAME = "crag_documents"

# IMPORTANT:
# This MUST be the same embedding model used in build_vector_db.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# Load Vector Database
# ============================================================

def load_vector_database():
    """
    Load the existing persistent ChromaDB vector database.
    """

    client = chromadb.PersistentClient(
        path=VECTORSTORE_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("=" * 50)
    print("Vector database loaded")
    print("=" * 50)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Total vectors : {collection.count()}")

    return collection


# ============================================================
# Load Embedding Model
# ============================================================

def load_embedding_model():
    """
    Load the same embedding model used during indexing.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(f"Embedding model: {EMBEDDING_MODEL}")

    return model


# ============================================================
# Retrieve Documents
# ============================================================

def retrieve(query, collection, embedding_model, top_k=TOP_K):
    """
    Retrieve the top-K most relevant chunks for a query.
    """

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# ============================================================
# Display Results
# ============================================================

def display_results(query, results):
    """
    Display retrieved chunks and their metadata.
    """

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No documents retrieved.")
        return

    for i, document in enumerate(documents):

        print("\n" + "-" * 70)
        print(f"RESULT {i + 1}")
        print("-" * 70)

        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None

        print(f"Source   : {metadata.get('source', 'Unknown')}")
        print(f"Page     : {metadata.get('page', 'Unknown')}")
        print(f"Distance : {distance}")

        print("\nText:")
        print(document[:1000])

        if len(document) > 1000:
            print("\n[Text truncated for display]")


# ============================================================
# Main
# ============================================================

def main():

    print("\nStarting local retriever...\n")

    # Load ChromaDB
    collection = load_vector_database()

    # Load embedding model
    embedding_model = load_embedding_model()

    # Test query
    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:
        print("No query entered.")
        return

    # Retrieve relevant chunks
    results = retrieve(
        query=query,
        collection=collection,
        embedding_model=embedding_model,
        top_k=TOP_K
    )

    # Display results
    display_results(
        query=query,
        results=results
    )


if __name__ == "__main__":
    main()