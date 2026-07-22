"""
backend/embeddings.py
Local Vector Embedding Generator using FastEmbed (ONNX Runtime)
"""

from typing import List
from fastembed import TextEmbedding


class EmbeddingEngine:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the ONNX local embedding model.
        Model: BAAI/bge-small-en-v1.5
        Output: 384-dimensional dense vectors
        """
        print(f"Loading local embedding model: {model_name}...")
        # FastEmbed downloads and loads the ONNX model locally on CPU
        self.model = TextEmbedding(model_name=model_name)
        print("Embedding model loaded successfully.")

    def generate_vector(self, text: str) -> List[float]:
        """
        Converts a single string/query into a 384-float vector list.
        Execution Time: < 10ms on standard CPU.
        """
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def generate_batch_vectors(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of document chunks into vectors in a single batch pass.
        """
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]


# Singleton instance so the model loads once and is shared across the backend
embedding_engine = EmbeddingEngine()


# Quick local test script
if __name__ == "__main__":
    test_query = "What is the policy for emergency medical leave?"
    vector = embedding_engine.generate_vector(test_query)
    
    print("\n--- Local Embedding Test Output ---")
    print(f"Sample Query: '{test_query}'")
    print(f"Vector Dimensions: {len(vector)}")
    print(f"First 5 Float Values: {vector[:5]}")