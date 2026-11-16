import argparse
import asyncio
import os

from agentmem.response_generator import run_chat
from agentmem.vectordb import client, create_memory_collection


async def main(user_id: int) -> None:
    try:
        await create_memory_collection()
        await run_chat(user_id)
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentMem: persistent memory for AI agents")
    parser.add_argument("user_id", type=int, nargs="?", default=1)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY before starting a chat")
    try:
        asyncio.run(main(args.user_id))
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
