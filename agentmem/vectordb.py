import os
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Filter, ScoredPoint, VectorParams, models

from agentmem.generate_embeddings import EMBEDDING_DIMENSIONS

client = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
COLLECTION_NAME = "memories"


class EmbeddedMemory(BaseModel):
    point_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    embedding: list[float]


class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float


async def create_memory_collection() -> None:
    if not await client.collection_exists(COLLECTION_NAME):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.DOT),
        )

        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Collection created")
    else:
        print("Collection exists")


async def insert_memories(memories: list[EmbeddedMemory]) -> None:
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=memory.point_id,
                payload={
                    "user_id": memory.user_id,
                    "categories": memory.categories,
                    "memory_text": memory.memory_text,
                    "date": memory.date,
                },
                vector=memory.embedding,
            )
            for memory in memories
        ],
        wait=True,
    )


async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
) -> list[RetrievedMemory]:
    must_conditions: list[models.Condition] = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if categories:
        must_conditions.append(
            models.FieldCondition(key="categories", match=models.MatchAny(any=categories))
        )
    outs = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=Filter(must=must_conditions),
        score_threshold=0.1,
        limit=2,
    )

    return [convert_retrieved_records(point) for point in outs.points]


async def delete_user_records(user_id: int) -> None:
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )
        ),
        wait=True,
    )


async def delete_records(point_ids: list[str]) -> None:
    if not point_ids:
        return
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
        wait=True,
    )


def convert_retrieved_records(point: ScoredPoint) -> RetrievedMemory:
    return RetrievedMemory(
        point_id=str(point.id),
        user_id=point.payload["user_id"],
        memory_text=point.payload["memory_text"],
        categories=point.payload["categories"],
        date=point.payload["date"],
        score=point.score,
    )


async def get_all_categories(user_id: int) -> list[str]:
    """
    Uses Qdrant's facet feature to efficiently get all unique categories
    from the indexed 'categories' field.
    """
    facet_filter = Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]
    )

    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,
    )

    return [hit.value for hit in facet_result.hits]


def stringify_retrieved_point(retrieved_memory: RetrievedMemory) -> str:
    return (
        f"{retrieved_memory.memory_text} (Categories: {retrieved_memory.categories}) "
        f"Relevance: {retrieved_memory.score:.2f}"
    )
