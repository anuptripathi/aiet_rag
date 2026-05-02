"""
Online retrieval: optional domain hint, glossary expansion, dense (+ sparse RRF), rerank,
parent-chunk expand, light cross-ref merge.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)
from qdrant_client.http import models as qm
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    GLOSSARY_JSON,
    MAX_PARENT_EXPAND,
    PREFETCH_LIMIT,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANK_MODEL,
    RERANK_TOP_N,
    RRF_K,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_VECTOR_NAME,
    EMBEDDING_MODEL,
    USE_RERANKER,
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
_encoder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_glossary: dict[str, str] | None = None
_collection_hybrid: bool | None = None


def get_dense_model() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def get_reranker() -> CrossEncoder | None:
    global _reranker
    if not USE_RERANKER:
        return None
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def load_glossary() -> dict[str, str]:
    global _glossary
    if _glossary is not None:
        return _glossary
    if not os.path.isfile(GLOSSARY_JSON):
        _glossary = {}
        return _glossary
    with open(GLOSSARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = data.get("glossary") or data
    if not isinstance(g, dict):
        g = {}
    _glossary = {str(k): str(v) for k, v in g.items()}
    return _glossary


def collection_is_hybrid() -> bool:
    global _collection_hybrid
    if _collection_hybrid is not None:
        return _collection_hybrid
    try:
        info = client.get_collection(COLLECTION_NAME)
    except Exception:
        _collection_hybrid = False
        return _collection_hybrid
    sp = info.config.params.sparse_vectors
    if sp is None:
        _collection_hybrid = False
    else:
        if isinstance(sp, dict):
            _collection_hybrid = len(sp) > 0
        else:
            _collection_hybrid = bool(sp)
    return _collection_hybrid


# --- query understanding ---

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "RAN": ["rrc", "pdcp", "rlc", "mac", "gnodeb", "gnb", "ue", "38.331", "38.300", "ng-ran"],
    "Core": ["amf", "smf", "upf", "pcf", "ausf", "nrf", "nssf", "udm", "23.501", "5gc"],
    "NAS": ["nas", "mm", "24.501", "registration", "service request"],
}


def classify_query(q: str) -> str | None:
    low = q.lower()
    scores: dict[str, int] = {d: 0 for d in DOMAIN_KEYWORDS}
    for dom, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in low:
                scores[dom] += 1
    best = max(scores.values())
    if best == 0:
        return None
    for dom, s in scores.items():
        if s == best:
            return dom
    return None


def expand_query_glossary(q: str, glossary: dict[str, str]) -> str:
    if not glossary:
        return q
    extra: list[str] = []
    for abbr, expl in sorted(glossary.items(), key=lambda x: -len(x[0])):
        if len(abbr) < 2:
            continue
        pat = r"(?<![A-Za-z0-9])" + re.escape(abbr) + r"(?![A-Za-z0-9])"
        if re.search(pat, q):
            extra.append(f"{expl}")
    if not extra:
        return q
    return q + " " + " ".join(dict.fromkeys(extra))


def _embed_sparse_query(text: str) -> SparseVector:
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
    emb = next(model.embed([text]))
    idx, val = emb.indices, emb.values
    if hasattr(idx, "tolist"):
        idx = idx.tolist()
    if hasattr(val, "tolist"):
        val = val.tolist()
    return SparseVector(
        indices=[int(i) for i in idx],
        values=[float(x) for x in val],
    )


@dataclass
class Hit:
    id: str
    score: float
    text: str
    chunk_id: str
    spec: str | None
    domain: str | None
    parent_id: str | None
    extra: dict[str, Any] = field(default_factory=dict)


def _to_hits(points) -> list[Hit]:
    hits: list[Hit] = []
    for p in points:
        pl = p.payload or {}
        hits.append(
            Hit(
                id=str(p.id),
                score=float(p.score or 0.0),
                text=pl.get("text") or "",
                chunk_id=pl.get("chunk_id") or "",
                spec=pl.get("spec"),
                domain=pl.get("domain"),
                parent_id=pl.get("parent_id"),
            )
        )
    return hits


def _vector_search(
    q_dense: list[float],
    q_sparse: SparseVector | None,
    limit: int,
    query_filter: Filter | None,
) -> list[Hit]:
    hybrid = collection_is_hybrid()
    if hybrid and q_sparse is not None:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(
                    query=qm.NearestQuery(nearest=q_dense),
                    using=DENSE_VECTOR_NAME,
                    limit=PREFETCH_LIMIT,
                    filter=query_filter,
                ),
                qm.Prefetch(
                    query=qm.NearestQuery(nearest=q_sparse),
                    using=SPARSE_VECTOR_NAME,
                    limit=PREFETCH_LIMIT,
                    filter=query_filter,
                ),
            ],
            query=qm.RrfQuery(rrf=qm.Rrf(k=RRF_K)),
            limit=limit,
        )
    elif hybrid:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_dense,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
        )
    else:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_dense,
            limit=limit,
            query_filter=query_filter,
        )
    return _to_hits(response.points)


def _rerank_hits(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    rer = get_reranker()
    if rer is None or len(hits) <= 1:
        return hits[:top_k]
    pairs = [(query, h.text[:8000]) for h in hits]
    scores = rer.predict(pairs)
    for h, s in zip(hits, scores):
        h.extra["rerank_score"] = float(s)
    ranked = sorted(hits, key=lambda x: x.extra.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_k]


def _parent_expand(hits: list[Hit], budget: int) -> list[Hit]:
    parents: list[str] = []
    for h in hits:
        if h.parent_id and h.parent_id not in parents:
            parents.append(h.parent_id)
        if len(parents) >= budget:
            break

    seen: set[str] = {h.chunk_id for h in hits}
    out: list[Hit] = list(hits)
    for pid in parents:
        flt = Filter(
            must=[FieldCondition(key="parent_id", match=MatchValue(value=pid))]
        )
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=64,
            with_payload=True,
        )
        for rec in records:
            pl = rec.payload or {}
            cid = pl.get("chunk_id") or ""
            if cid and cid not in seen:
                seen.add(cid)
                out.append(
                    Hit(
                        id=str(rec.id),
                        score=0.0,
                        text=pl.get("text") or "",
                        chunk_id=cid,
                        spec=pl.get("spec"),
                        domain=pl.get("domain"),
                        parent_id=pl.get("parent_id"),
                        extra={"parent_expand": True},
                    )
                )

    out.sort(key=lambda x: (x.spec or "", x.chunk_id))
    return out


def retrieve(
    query: str,
    top_k: int = 5,
    *,
    use_glossary: bool = True,
    use_domain_filter: bool = False,
    expand_parents: bool = True,
    cross_ref_second_pass: bool = True,
) -> dict[str, Any]:
    """
    Full retrieval stack. Returns dict with hits and diagnostics.
    """
    glossary = load_glossary() if use_glossary else {}
    expanded_q = expand_query_glossary(query, glossary) if use_glossary else query
    dom = classify_query(query)
    q_filter: Filter | None = None
    if use_domain_filter and dom:
        q_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=dom))]
        )

    model = get_dense_model()
    q_dense = model.encode(
        expanded_q, normalize_embeddings=True
    ).tolist()

    q_sparse: SparseVector | None = None
    if collection_is_hybrid():
        try:
            q_sparse = _embed_sparse_query(expanded_q)
        except ImportError:
            q_sparse = None

    pool = max(RERANK_TOP_N, top_k * 4, PREFETCH_LIMIT // 2)
    hits = _vector_search(q_dense, q_sparse, limit=pool, query_filter=q_filter)

    if cross_ref_second_pass and use_glossary and expanded_q.strip() != query.strip():
        qd2 = model.encode(query, normalize_embeddings=True).tolist()
        qs2: SparseVector | None = None
        if collection_is_hybrid():
            try:
                qs2 = _embed_sparse_query(query)
            except ImportError:
                qs2 = None
        h2 = _vector_search(qd2, qs2, limit=max(pool // 2, top_k * 2), query_filter=q_filter)
        merged: dict[str, Hit] = {h.chunk_id: h for h in hits}
        for h in h2:
            merged.setdefault(h.chunk_id, h)
        hits = sorted(merged.values(), key=lambda x: -x.score)

    hits = _rerank_hits(expanded_q, hits, top_k=min(len(hits), RERANK_TOP_N))
    primary = hits[:top_k]

    if expand_parents:
        primary = _parent_expand(primary, MAX_PARENT_EXPAND)

    return {
        "query": query,
        "expanded_query": expanded_q,
        "domain_guess": dom,
        "hybrid": collection_is_hybrid(),
        "hits": [
            {
                "score": h.score,
                "text": h.text,
                "spec": h.spec,
                "domain": h.domain,
                "chunk_id": h.chunk_id,
                "parent_id": h.parent_id,
                **h.extra,
            }
            for h in primary
        ],
    }


def legacy_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Shape compatible with early retrieve.py CLI."""
    out = retrieve(query, top_k=top_k, cross_ref_second_pass=False, expand_parents=False)
    return [
        {
            "score": h["score"],
            "text": h["text"],
            "spec": h["spec"],
            "domain": h["domain"],
            "chunk_id": h["chunk_id"],
        }
        for h in out["hits"][:top_k]
    ]
