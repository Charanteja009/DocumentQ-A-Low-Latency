"""
backend/vector_store.py
Qdrant Vector DB Manager with Session-Based Multi-Tenancy Filtering
"""

import os
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "rag_documents"
VECTOR_SIZE = 384  # Matches BAAI/bge-small-en-v1.5 output dimension


class VectorStoreManager:
    def __init__(self):
        # Reads connection info from environment variables (local or Qdrant Cloud)
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

        print(f"Connecting to Qdrant at: {qdrant_url}...")
        self.client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    async def init_collection(self):
        """
        Creates the 'rag_documents' collection if it does not exist yet
        and builds a KEYWORD payload index on 'session_id' for fast filtering.
        """
        collections_response = await self.client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]

        if COLLECTION_NAME not in existing_collections:
            print(f"Creating collection '{COLLECTION_NAME}'...")
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{COLLECTION_NAME}' created successfully.")

        # Create payload index for session_id (required by Qdrant Cloud)
        print("Ensuring payload index on 'session_id'...")
        await self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="session_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("Payload index on 'session_id' ready.")
        
    async def upsert_chunks(
        self, 
        session_id: str, 
        chunks: List[str], 
        vectors: List[List[float]], 
        doc_name: str
    ):
        """
        Inserts document chunks + vectors into Qdrant.
        Attaches session_id to every point's payload for multi-tenant isolation.
        """
        points = [
            models.PointStruct(
                id=hash(f"{session_id}_{doc_name}_{i}"),
                vector=vector,
                payload={
                    "session_id": session_id,
                    "text": chunk,
                    "doc_name": doc_name,
                    "chunk_index": i
                }
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        await self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Successfully upserted {len(points)} chunks for session: {session_id}")

    async def search_similar_chunks(
        self, 
        session_id: str, 
        query_vector: List[float], 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executes vector search with strict session-level payload filtering.
        Guarantees that User A can NEVER retrieve User B's document chunks.
        """
        search_result = await self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            # Mandatory Payload Filter
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            ),
            limit=top_k
        )

        # Extract text payloads and scores from search results
        return [
            {
                "text": hit.payload["text"],
                "doc_name": hit.payload["doc_name"],
                "score": hit.score
            }
            for hit in search_result
        ]


# Global instance
vector_store = VectorStoreManager()


# Test block
if __name__ == "__main__":
    import asyncio

    async def test_run():
        await vector_store.init_collection()

    asyncio.run(test_run())