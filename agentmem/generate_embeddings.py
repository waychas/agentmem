from openai import AsyncOpenAI

EMBEDDING_DIMENSIONS = 64


async def generate_embeddings(strings: list[str]) -> list[list[float]]:
    async with AsyncOpenAI() as client:
        response = await client.embeddings.create(
            input=strings,
            model="text-embedding-3-small",
            dimensions=EMBEDDING_DIMENSIONS,
        )
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
