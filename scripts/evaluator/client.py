"""Async HTTP client for evaluation, with built-in latency tracking."""
import time
import httpx
from typing import Optional

API_BASE = "http://127.0.0.1:8002"

class AsyncEvalClient:
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.latencies: list[float] = []

    async def chat(self, messages: list[dict[str, str]]) -> tuple[Optional[dict], float]:
        """
        Sends a chat request and returns the parsed JSON and latency in ms.
        Returns (None, latency) if request fails or times out.
        """
        start = time.perf_counter()
        try:
            response = await self.client.post(
                f"{API_BASE}/chat",
                json={"messages": messages}
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.latencies.append(elapsed_ms)
            
            if response.status_code == 200:
                return response.json(), elapsed_ms
            else:
                return None, elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.latencies.append(elapsed_ms)
            return None, elapsed_ms

    async def close(self):
        await self.client.aclose()
