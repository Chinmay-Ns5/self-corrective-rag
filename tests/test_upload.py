from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import chromadb

from src.ingestion.upload import add_pdf
from src.retrieval.hybrid import HybridRetriever


class FakeEmbedding:
    def encode(self, value, show_progress_bar=False):
        vectors = [[0.1, 0.2] for _ in value] if isinstance(value, list) else [0.1, 0.2]
        return SimpleNamespace(tolist=lambda: vectors)


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class UploadTests(unittest.TestCase):
    def test_uploaded_pdf_is_saved_and_available_to_retrieval(self):
        collection = chromadb.EphemeralClient().create_collection(name=f"upload_test_{uuid.uuid4().hex}")
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.ingestion.upload.PdfReader",
                      return_value=SimpleNamespace(pages=[FakePage("Zephyr uses an orbiting mirror.")])):
            result = add_pdf("new_study.pdf", b"%PDF-test", collection=collection,
                             embedding_model=FakeEmbedding(), pdf_dir=Path(directory))
            self.assertEqual(result.filename, "new_study.pdf")
            self.assertEqual(result.pages, 1)
            self.assertEqual(result.chunks, 1)
            self.assertEqual((Path(directory) / result.filename).read_bytes(), b"%PDF-test")
            found = HybridRetriever(collection, FakeEmbedding()).retrieve("What does Zephyr use?", 1)
            self.assertIn("orbiting mirror", found["documents"][0][0])
            self.assertEqual(found["metadatas"][0][0]["source"], result.filename)

    def test_duplicate_name_is_rejected_without_changing_index(self):
        collection = chromadb.EphemeralClient().create_collection(name=f"upload_test_{uuid.uuid4().hex}")
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.ingestion.upload.PdfReader",
                      return_value=SimpleNamespace(pages=[FakePage("Extractable text")])):
            add_pdf("example.pdf", b"%PDF-test", collection=collection,
                    embedding_model=FakeEmbedding(), pdf_dir=Path(directory))
            with self.assertRaisesRegex(ValueError, "already in the knowledge base"):
                add_pdf("../example.pdf", b"%PDF-other", collection=collection,
                        embedding_model=FakeEmbedding(), pdf_dir=Path(directory))
            self.assertEqual(collection.count(), 1)

    def test_pdf_without_extractable_text_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.ingestion.upload.PdfReader",
                      return_value=SimpleNamespace(pages=[FakePage("   ")])):
            with self.assertRaisesRegex(ValueError, "no extractable text"):
                add_pdf("scan.pdf", b"%PDF-test", pdf_dir=Path(directory))
            self.assertFalse((Path(directory) / "scan.pdf").exists())


if __name__ == "__main__":
    unittest.main()
