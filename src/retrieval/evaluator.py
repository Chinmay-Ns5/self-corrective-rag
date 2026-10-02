import os
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initial thresholds from the project implementation plan
HIGH_THRESHOLD = 0.75
LOW_THRESHOLD = 0.40


# ============================================================
# Load Embedding Model
# ============================================================

def load_embedding_model():
    """
    Load the same embedding model used by the retriever.
    """

    print("\nLoading evaluator embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding model: {EMBEDDING_MODEL}")

    return model


# ============================================================
# Calculate Semantic Similarity
# ============================================================

def calculate_similarity(query, document, model):
    """
    Calculate cosine similarity between the query and
    a retrieved document.
    """

    query_embedding = model.encode(query)
    document_embedding = model.encode(document)

    query_embedding = np.array(query_embedding)
    document_embedding = np.array(document_embedding)

    # Cosine similarity
    similarity = np.dot(
        query_embedding,
        document_embedding
    ) / (
        np.linalg.norm(query_embedding)
        * np.linalg.norm(document_embedding)
    )

    return float(similarity)


# ============================================================
# Evaluate Retrieved Documents
# ============================================================

def evaluate_retrieval(query, documents, model):
    """
    Evaluate how relevant the retrieved documents are
    to the user's query.

    Returns:

    {
        "label": "CORRECT",
        "score": 0.91,
        "reason": "...",
        "document_scores": [...]
    }
    """

    if not documents:
        return {
            "label": "INCORRECT",
            "score": 0.0,
            "reason": "No documents were retrieved.",
            "document_scores": []
        }

    scores = []

    # Calculate similarity for every retrieved chunk
    for document in documents:

        score = calculate_similarity(
            query=query,
            document=document,
            model=model
        )

        scores.append(score)

    # Use the average relevance of the retrieved context
    overall_score = float(np.mean(scores))

    # Determine relevance category
    if overall_score >= HIGH_THRESHOLD:

        label = "CORRECT"

        reason = (
            "Retrieved context is sufficiently relevant "
            "to the query."
        )

    elif overall_score >= LOW_THRESHOLD:

        label = "AMBIGUOUS"

        reason = (
            "Retrieved context is related to the query "
            "but may be incomplete or uncertain."
        )

    else:

        label = "INCORRECT"

        reason = (
            "Retrieved context is not sufficiently relevant "
            "to the query."
        )

    return {
        "label": label,
        "score": overall_score,
        "reason": reason,
        "document_scores": scores
    }


# ============================================================
# Display Evaluation
# ============================================================

def display_evaluation(query, evaluation):
    """
    Display the evaluator result in a readable format.
    """

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\nQuery:")
    print(query)

    print("\nLabel:")
    print(evaluation["label"])

    print("\nScore:")
    print(f"{evaluation['score']:.4f}")

    print("\nReason:")
    print(evaluation["reason"])

    print("\nIndividual document scores:")

    for i, score in enumerate(
        evaluation["document_scores"],
        start=1
    ):
        print(f"  Document {i}: {score:.4f}")

    print("\nThresholds:")
    print(f"  HIGH_THRESHOLD = {HIGH_THRESHOLD}")
    print(f"  LOW_THRESHOLD  = {LOW_THRESHOLD}")


# ============================================================
# Main Test
# ============================================================

def main():

    print("\nStarting retrieval evaluator...\n")

    # Import your existing retriever functions
    from retriever import (
        load_vector_database,
        retrieve
    )

    # Load ChromaDB
    collection = load_vector_database()

    # Load embedding model
    embedding_model = load_embedding_model()

    # Ask for query
    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:
        print("No query entered.")
        return

    # Retrieve documents using YOUR existing retriever
    results = retrieve(
        query=query,
        collection=collection,
        embedding_model=embedding_model,
        top_k=5
    )

    # Extract retrieved documents
    documents = results.get(
        "documents",
        [[]]
    )[0]

    if not documents:
        print("\nNo documents retrieved.")
        return

    print(
        f"\nEvaluating {len(documents)} retrieved documents..."
    )

    # Evaluate retrieved context
    evaluation = evaluate_retrieval(
        query=query,
        documents=documents,
        model=embedding_model
    )

    # Display result
    display_evaluation(
        query=query,
        evaluation=evaluation
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()