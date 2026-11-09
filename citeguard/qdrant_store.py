from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from citeguard.schemas import MetadataFilter

GRADE_ALLOWED_MAP: dict[str, list[str]] = {
    "A": ["A"],
    "B": ["A", "B"],
    "C": ["A", "B", "C"],
}


@dataclass(frozen=True)
class QdrantPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class QdrantScoredHit:
    point_id: str
    chunk_uid: str
    score: float
    payload: dict[str, Any]


class InMemoryQdrantTransport:
    """In-memory tracking adapter for test introspection."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, QdrantPoint]] = {}


class QdrantVectorStore:
    """Async Qdrant client wrapper supporting collection init, payload filtering, upsert, and deletion."""

    def __init__(
        self,
        base_url: str | None = None,
        collection_name: str | None = None,
        timeout_seconds: float = 15.0,
        transport: Any = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("CITEGUARD_QDRANT_URL") or "http://localhost:6333"
        ).rstrip("/")
        self.collection_name = (
            collection_name or os.getenv("CITEGUARD_QDRANT_COLLECTION") or "citeguard_chunks"
        )
        self._timeout = timeout_seconds
        self._transport = transport
        if client is not None:
            self._client = client
        elif transport is not None or self.base_url in {":memory:", "memory"}:
            self._client = AsyncQdrantClient(location=":memory:")
        else:
            self._client = AsyncQdrantClient(url=self.base_url, timeout=int(self._timeout))
        self._collection_ready = False

    async def ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        collections = await self._client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        if not exists:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        self._collection_ready = True

    async def upsert_points(self, points: list[QdrantPoint], vector_size: int) -> int:
        if not points:
            return 0
        await self.ensure_collection(vector_size)
        q_points = [
            PointStruct(id=pt.point_id, vector=pt.vector, payload=pt.payload)
            for pt in points
        ]
        await self._client.upsert(collection_name=self.collection_name, points=q_points)
        if isinstance(self._transport, InMemoryQdrantTransport):
            bucket = self._transport.points.setdefault(self.collection_name, {})
            for pt in points:
                bucket[pt.point_id] = pt
        return len(points)

    async def delete_points(self, point_ids: list[str]) -> int:
        if not point_ids:
            return 0
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=cast(list[Any], point_ids)),
        )
        if isinstance(self._transport, InMemoryQdrantTransport):
            bucket = self._transport.points.setdefault(self.collection_name, {})
            for pid in point_ids:
                bucket.pop(pid, None)
        return len(point_ids)

    async def query_similar(
        self,
        query_vector: list[float],
        filters: MetadataFilter,
        limit: int = 10,
    ) -> list[QdrantScoredHit]:
        await self.ensure_collection(len(query_vector))
        must: list[Any] = []
        if filters.category:
            must.append(FieldCondition(key="category", match=MatchValue(value=filters.category)))
        if filters.language:
            must.append(FieldCondition(key="language", match=MatchValue(value=filters.language)))
        if filters.doc_key:
            must.append(FieldCondition(key="doc_key", match=MatchValue(value=filters.doc_key)))
        if filters.min_source_grade:
            allowed = GRADE_ALLOWED_MAP.get(filters.min_source_grade, ["A", "B", "C"])
            must.append(FieldCondition(key="source_grade", match=MatchAny(any=allowed)))

        q_filter = Filter(must=must) if must else None
        res = await self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[QdrantScoredHit] = []
        for p in res.points:
            payload = p.payload or {}
            if (
                filters.effective_after
                and str(payload.get("effective_date", "")) < filters.effective_after
            ):
                continue
            hits.append(
                QdrantScoredHit(
                    point_id=str(p.id),
                    chunk_uid=str(payload.get("chunk_uid", "")),
                    score=float(p.score),
                    payload=payload,
                )
            )
        return hits
