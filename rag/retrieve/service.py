"""
FastAPI retrieval service: POST /retrieve, GET /healthz
"""

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.config import RETRIEVE_BIND_HOST, RETRIEVE_PORT
from rag.retrieve.pipeline import retrieve

app = FastAPI(title="Telecom RAG Retrieve", version="0.1.0")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    use_glossary: bool = True
    use_domain_filter: bool = False
    expand_parents: bool = True
    cross_ref_second_pass: bool = True


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(body: RetrieveRequest) -> dict[str, Any]:
    return retrieve(
        body.query,
        top_k=body.top_k,
        use_glossary=body.use_glossary,
        use_domain_filter=body.use_domain_filter,
        expand_parents=body.expand_parents,
        cross_ref_second_pass=body.cross_ref_second_pass,
    )


def main():
    import uvicorn

    uvicorn.run(
        "rag.retrieve.service:app",
        host=RETRIEVE_BIND_HOST,
        port=RETRIEVE_PORT,
        reload=os.environ.get("RETRIEVE_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
