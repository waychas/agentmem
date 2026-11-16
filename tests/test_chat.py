import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import main
from agentmem import generate_embeddings, response_generator


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_blank_input_is_skipped_and_history_is_bounded(self):
        history_sizes = []

        async def respond(**kwargs):
            history_sizes.append(len(kwargs["transcript"]))
            return SimpleNamespace(response="Hello", save_memory=False)

        agent = SimpleNamespace(acall=AsyncMock(side_effect=respond))
        console = MagicMock()
        console.input.side_effect = [""] + ["Hello"] * 12 + [EOFError]
        with (
            patch.object(response_generator, "console", console),
            patch.object(response_generator.dspy, "ReAct", return_value=agent),
            patch.object(response_generator, "get_all_categories", AsyncMock(return_value=[])),
            patch.object(response_generator, "update_memories", new_callable=AsyncMock) as update,
        ):
            with self.assertRaises(EOFError):
                await response_generator.run_chat(1)
        self.assertEqual(len(history_sizes), 12)
        self.assertEqual(history_sizes, [min(turn * 2, 20) for turn in range(12)])
        update.assert_not_awaited()

    async def test_memory_failure_is_reported_without_losing_response(self):
        console = MagicMock()
        console.input.side_effect = ["I moved", EOFError]
        agent = SimpleNamespace(
            acall=AsyncMock(return_value=SimpleNamespace(response="Understood", save_memory=True))
        )
        with (
            patch.object(response_generator, "console", console),
            patch.object(response_generator.dspy, "ReAct", return_value=agent),
            patch.object(response_generator, "get_all_categories", AsyncMock(return_value=[])),
            patch.object(response_generator, "update_memories", AsyncMock(side_effect=RuntimeError("Offline"))),
        ):
            with self.assertRaises(EOFError):
                await response_generator.run_chat(1)
        console.print.assert_any_call("Memory maintenance failed: Offline", markup=False)
        console.print.assert_any_call("\nAI: Understood\n\n", style="bold green")

    async def test_successful_memory_update_refreshes_categories(self):
        console = MagicMock()
        console.input.side_effect = ["I moved", EOFError]
        agent = SimpleNamespace(
            acall=AsyncMock(return_value=SimpleNamespace(response="Understood", save_memory=True))
        )
        with (
            patch.object(response_generator, "console", console),
            patch.object(response_generator.dspy, "ReAct", return_value=agent),
            patch.object(response_generator, "get_all_categories", AsyncMock(return_value=[])) as categories,
            patch.object(response_generator, "update_memories", AsyncMock(return_value="Saved")) as update,
        ):
            with self.assertRaises(EOFError):
                await response_generator.run_chat(1)
        self.assertEqual(categories.await_count, 2)
        update.assert_awaited_once_with(
            user_id=1,
            messages=[{"role": "user", "content": "I moved"}, {"role": "assistant", "content": "Understood"}],
        )

    async def test_startup_initializes_collection_and_closes_client(self):
        with (
            patch.object(main, "create_memory_collection", new_callable=AsyncMock) as initialize,
            patch.object(main, "run_chat", new_callable=AsyncMock) as chat,
            patch.object(main.client, "close", new_callable=AsyncMock) as close,
        ):
            await main.main(42)
        initialize.assert_awaited_once()
        chat.assert_awaited_once_with(42)
        close.assert_awaited_once()

    async def test_startup_failure_still_closes_client(self):
        with (
            patch.object(main, "create_memory_collection", AsyncMock(side_effect=RuntimeError("Offline"))),
            patch.object(main, "run_chat", new_callable=AsyncMock) as chat,
            patch.object(main.client, "close", new_callable=AsyncMock) as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "Offline"):
                await main.main(1)
        chat.assert_not_awaited()
        close.assert_awaited_once()

    async def test_embeddings_preserve_input_order_and_close_client(self):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.embeddings.create = AsyncMock(return_value=SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]))
        with patch.object(generate_embeddings, "AsyncOpenAI", return_value=client):
            result = await generate_embeddings.generate_embeddings(["First", "Second"])
        self.assertEqual(result, [[1.0, 0.0], [0.0, 1.0]])
        client.embeddings.create.assert_awaited_once_with(
            input=["First", "Second"], model="text-embedding-3-small", dimensions=64
        )
        client.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
