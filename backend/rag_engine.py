"""
backend/rag_engine.py
Core RAG Pipeline: Vector Search Retrieval + Async Token Streaming via Groq
"""

import os
from typing import AsyncGenerator
from groq import AsyncGroq
from dotenv import load_dotenv

from embeddings import embedding_engine
from vector_store import vector_store

load_dotenv()


class RAGEngine:
    def __init__(self):
        # Initialize the Async Groq client for low-latency streaming
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.client = AsyncGroq(api_key=groq_api_key)
        self.model_name = "llama-3.1-8b-instant"  # Optimized for ultra-fast TTFT

    async def generate_rag_stream(
        self, 
        session_id: str, 
        user_prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        1. Embed user query locally (<10ms).
        2. Perform session-filtered vector search in Qdrant (<10ms).
        3. Construct augmented prompt with retrieved chunks.
        4. Yield tokens from Groq LLM in real-time.
        """
        # Step 1: Embed Query
        query_vector = embedding_engine.generate_vector(user_prompt)

        # Step 2: Retrieve Relevant Context (Session-Isolated)
        retrieved_chunks = await vector_store.search_similar_chunks(
            session_id=session_id,
            query_vector=query_vector,
            top_k=3
        )

        # Format Context
        if retrieved_chunks:
            context_text = "\n\n".join(
                [f"[Doc: {chunk['doc_name']}]\n{chunk['text']}" for chunk in retrieved_chunks]
            )
        else:
            context_text = "No prior document context found for this session."

        # Step 3: Construct System & User Prompts
        system_prompt = (
            "You are an intelligent, low-latency document assistant. "
            "Answer the user's question accurately using ONLY the provided context below. "
            "If the answer cannot be found in the context, state clearly that information is unavailable."
        )

        user_content = f"Context:\n{context_text}\n\nQuestion:\n{user_prompt}"

        # Step 4: Stream Tokens from Groq LLM
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True  # Enables Server-Sent Token Streaming
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


# Global Singleton Instance
rag_engine = RAGEngine()