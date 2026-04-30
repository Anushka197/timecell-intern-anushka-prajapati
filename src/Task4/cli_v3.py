import asyncio
import traceback
from ingest_v3 import PortfolioIngestorV3
from engine_v3 import build_agent, query_with_critic


async def async_main():
    print("Initializing Local Sovereign AI...")
    try:
        ingestor = PortfolioIngestorV3()
        index = ingestor.load_index()
        agent = build_agent(index)

        print("\n✅ Local Agent Ready (Offline). Type 'exit' to quit.")

        while True:
            query = input("\nUser ❯ ").strip()
            if query.lower() == "exit":
                break

            print("\nThinking locally...\n")

            response = await query_with_critic(query, agent)
            print(f"\nAI ❯ {response}")

            response = response if isinstance(response, str) else response.response
            print(f"\nAI ❯ {response}")

    except Exception:
        print("\n❌ CRITICAL SYSTEM ERROR:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(async_main())