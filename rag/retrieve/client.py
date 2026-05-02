"""
Thin HTTP client for the retrieval service (/retrieve, /healthz).
"""

from typing import Any

import httpx

from rag.config import RETRIEVE_BASE_URL


class RetrieveClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 120.0):
        self.base_url = (base_url or RETRIEVE_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s

    def healthz(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_s) as c:
            r = c.get(f"{self.base_url}/healthz")
            r.raise_for_status()
            return r.json()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        use_glossary: bool = True,
        use_domain_filter: bool = False,
        expand_parents: bool = True,
        cross_ref_second_pass: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "top_k": top_k,
            "use_glossary": use_glossary,
            "use_domain_filter": use_domain_filter,
            "expand_parents": expand_parents,
            "cross_ref_second_pass": cross_ref_second_pass,
        }
        with httpx.Client(timeout=self.timeout_s) as c:
            r = c.post(f"{self.base_url}/retrieve", json=payload)
            r.raise_for_status()
            return r.json()
