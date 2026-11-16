import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from qdrant_client import AsyncQdrantClient

from agentmem import update_memory, vectordb

VECTOR = [1.0] + [0.0] * 63


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncQdrantClient(location=":memory:")
        self.client_patch = patch.object(vectordb, "client", self.client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.addAsyncCleanup(self.client.close)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
            await vectordb.create_memory_collection()
        self.memory = vectordb.EmbeddedMemory(
            user_id=1,
            memory_text="I live in Bangalore",
            categories=["location"],
            date="2026-01-01",
            embedding=VECTOR,
        )
        self.other = vectordb.EmbeddedMemory(
            user_id=2,
            memory_text="I like tea",
            categories=["food"],
            date="2026-01-01",
            embedding=VECTOR,
        )
        await vectordb.insert_memories([self.memory, self.other])

    async def run_tool(self, name, *args, fail_embedding=False):
        existing = await vectordb.search_memories(VECTOR, user_id=1)

        async def act(**kwargs):
            tools = constructor.call_args.kwargs["tools"]
            tool = next(tool for tool in tools if tool.__name__ == name)
            await tool(*args)
            return SimpleNamespace(summary="Done")

        agent = SimpleNamespace(acall=AsyncMock(side_effect=act))
        embedding = AsyncMock(return_value=[VECTOR])
        if fail_embedding:
            embedding.side_effect = RuntimeError("Embedding failed")
        with (
            patch.object(update_memory.dspy, "ReAct", return_value=agent) as constructor,
            patch.object(update_memory, "generate_embeddings", embedding),
        ):
            return await update_memory.update_memories_agent(
                1, [{"role": "user", "content": "I moved"}], existing
            )

    async def test_search_is_scoped_by_user_and_category(self):
        results = await vectordb.search_memories(VECTOR, 1, ["location"])
        self.assertEqual([memory.point_id for memory in results], [self.memory.point_id])
        self.assertEqual(await vectordb.search_memories(VECTOR, 1, ["food"]), [])
        results = await vectordb.search_memories(VECTOR, 2, [])
        self.assertEqual([memory.point_id for memory in results], [self.other.point_id])

    async def test_update_preserves_record_id_and_replaces_fact(self):
        await self.run_tool("update", 0, "I live in Hyderabad", ["location"])
        results = await vectordb.search_memories(VECTOR, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].point_id, self.memory.point_id)
        self.assertEqual(results[0].memory_text, "I live in Hyderabad")
        self.assertEqual((await vectordb.search_memories(VECTOR, 2))[0].memory_text, "I like tea")

    async def test_failed_embedding_keeps_original_memory(self):
        with self.assertRaisesRegex(RuntimeError, "Embedding failed"):
            await self.run_tool("update", 0, "New fact", [], fail_embedding=True)
        results = await vectordb.search_memories(VECTOR, 1)
        self.assertEqual(results[0].memory_text, self.memory.memory_text)
        self.assertEqual(results[0].point_id, self.memory.point_id)

    async def test_failed_write_keeps_original_memory(self):
        with patch.object(update_memory, "insert_memories", AsyncMock(side_effect=RuntimeError("Write failed"))):
            with self.assertRaisesRegex(RuntimeError, "Write failed"):
                await self.run_tool("update", 0, "New fact", [])
        self.assertEqual((await vectordb.search_memories(VECTOR, 1))[0].memory_text, self.memory.memory_text)

    async def test_delete_resolves_agent_index_to_record_id(self):
        await self.run_tool("delete", [0])
        self.assertEqual(await vectordb.search_memories(VECTOR, 1), [])
        self.assertEqual((await vectordb.search_memories(VECTOR, 2))[0].point_id, self.other.point_id)

    async def test_invalid_ids_are_rejected_before_any_deletion(self):
        for invalid in (-1, 1):
            with self.subTest(memory_id=invalid):
                with self.assertRaisesRegex(ValueError, "Unknown memory ID"):
                    await self.run_tool("delete", [0, invalid])
                with self.assertRaisesRegex(ValueError, "Unknown memory ID"):
                    await self.run_tool("update", invalid, "New fact", [])
        self.assertEqual(len(await vectordb.search_memories(VECTOR, 1)), 1)

    async def test_rejects_another_users_memories(self):
        memories = await vectordb.search_memories(VECTOR, 2)
        with self.assertRaisesRegex(ValueError, "another user's"):
            await update_memory.update_memories_agent(1, [], memories)

    async def test_add_and_noop(self):
        await self.run_tool("add_memory", "I like coffee", ["food"])
        self.assertEqual(len(await vectordb.search_memories(VECTOR, 1)), 2)
        self.assertEqual(await self.run_tool("noop"), "Done")
        self.assertEqual(len(await vectordb.search_memories(VECTOR, 1)), 2)

    async def test_delete_all_is_user_scoped(self):
        await vectordb.delete_user_records(1)
        self.assertEqual(await vectordb.search_memories(VECTOR, 1), [])
        self.assertEqual(len(await vectordb.search_memories(VECTOR, 2)), 1)

    async def test_categories_are_scoped_and_only_queried_once(self):
        with patch.object(self.client, "facet", wraps=self.client.facet) as facet:
            self.assertEqual(await vectordb.get_all_categories(1), ["location"])
            facet.assert_awaited_once()
        self.assertEqual(await vectordb.get_all_categories(2), ["food"])

    async def test_collection_initialization_is_idempotent(self):
        await vectordb.create_memory_collection()
        self.assertEqual(len(await vectordb.search_memories(VECTOR, 1)), 1)

    async def test_empty_delete_does_not_call_database(self):
        with patch.object(self.client, "delete", new_callable=AsyncMock) as delete:
            await vectordb.delete_records([])
            delete.assert_not_awaited()

    async def test_missing_user_message_fails_before_embedding(self):
        with patch.object(update_memory, "generate_embeddings", new_callable=AsyncMock) as embed:
            for messages in ([], [{"role": "assistant", "content": "Hello"}], [{"role": "user", "content": ""}]):
                with self.subTest(messages=messages):
                    with self.assertRaisesRegex(ValueError, "non-empty user message"):
                        await update_memory.update_memories(1, messages)
            embed.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
