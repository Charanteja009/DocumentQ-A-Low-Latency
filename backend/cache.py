"""
backend/cache.py
Redis Async Caching Engine for Sub-30ms Response Times
"""

import os
import hashlib
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()


class RedisCacheManager:
    def __init__(self):
        # Reads connection info from env vars (Local Redis or Upstash Cloud Redis)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        print(f"Connecting to Redis at: {redis_url}...")
        self.client = aioredis.from_url(redis_url, decode_responses=True)

    def _generate_key(self, session_id: str, prompt: str) -> str:
        """
        Generates a namespaced MD5 cache key combining session_id and user prompt.
        Pattern: cache:{session_id}:{prompt_hash}
        """
        prompt_hash = hashlib.md5(prompt.strip().lower().encode("utf-8")).hexdigest()
        return f"cache:{session_id}:{prompt_hash}"

    async def get(self, session_id: str, prompt: str) -> str | None:
        """
        Retrieves a cached answer from Redis in RAM (<2ms).
        Returns string if hit, None if miss.
        """
        try:
            key = self._generate_key(session_id, prompt)
            cached_val = await self.client.get(key)
            if cached_val:
                print(f"⚡ [CACHE HIT] Serving response directly from Redis for key: {key}")
                return cached_val
            return None
        except Exception as e:
            print(f"⚠️ Redis read error (falling back to RAG pipeline): {e}")
            return None

    async def set(self, session_id: str, prompt: str, response_text: str, ttl_seconds: int = 86400):
        """
        Stores generated response in Redis with an expiration time (default: 24 hours).
        """
        try:
            key = self._generate_key(session_id, prompt)
            await self.client.set(key, response_text, ex=ttl_seconds)
            print(f"💾 [CACHE STORED] Saved key in Redis: {key}")
        except Exception as e:
            print(f"⚠️ Redis write error: {e}")

    async def append_history(self, session_id: str, role: str, text: str):
        """Appends a message (user or assistant) to the Redis session history list."""
        try:
            key = f"history:{session_id}"
            await self.client.rpush(key, f"{role}: {text}")
            await self.client.expire(key, 86400)  # History expires after 24 hrs
        except Exception as e:
            print(f"⚠️ Redis history write error: {e}")

    async def get_history(self, session_id: str) -> list[str]:
        """Retrieves all past conversation messages for a session."""
        try:
            key = f"history:{session_id}"
            return await self.client.lrange(key, 0, -1)
        except Exception as e:
            print(f"⚠️ Redis history read error: {e}")
            return []


# Global Singleton Instance
cache_manager = RedisCacheManager()


# Local test block
if __name__ == "__main__":
    import asyncio

    async def test_cache():
        session = "test_session_123"
        prompt = "What is the policy for leave?"
        
        # Test Cache Miss
        result = await cache_manager.get(session, prompt)
        print("Initial Check (Expected None):", result)
        
        # Store in Cache
        await cache_manager.set(session, prompt, "Leave policy grants 12 days annually.")
        
        # Test Cache Hit
        hit_result = await cache_manager.get(session, prompt)
        print("Second Check (Expected Answer):", hit_result)

    asyncio.run(test_cache())