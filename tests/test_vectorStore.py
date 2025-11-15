import os
import sys
import unittest

# --- Path setup to allow 'from src...' imports ---
# Add the project root to sys.path
print("current file", os.path.join(os.path.dirname(__file__), ".."))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, project_root)

from retriever.vectorStore import build_retriever_from_docs


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Build retriever (use a small chunk size for speed in tests)
        self.retriever = build_retriever_from_docs(
            embedding_model_name="google/embeddinggemma-300M",
            chunk_size=500,
            chunk_overlap=50,
            k=5,
        )

    def test_rag_response_quality(self):
        # Example queries to test RAG quality
        queries = [
            "What is the main topic of the documents?",
            "write an article on financial growth of mercedes benz between 2023 and 2024.",
            "Write an article on distribution of number of covid cases across different nations.",
            "What is the climate action plan of mercedes benz in 2025 and how is it different from 2024?",
        ]
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            print(f"\nQuery: {query}\nRetrieved Contexts:")
            for i, doc in enumerate(docs):
                print(
                    f"Context {i+1}:\n{doc.page_content[:500]}...\n"
                )  # print first 500 chars for brevity
            self.assertTrue(len(docs) > 0)
            self.assertTrue(all(hasattr(doc, "page_content") for doc in docs))


if __name__ == "__main__":
    unittest.main()
