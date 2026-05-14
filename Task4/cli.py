"""
cli.py — Command Line Interface for Portfolio Intelligence Engine
=================================================================
Provides an interactive terminal chat for querying Apple 10-K data.
"""

import sys
import logging
import warnings
import os

from ingest import PortfolioIngestor
# Import the new critic function
from engine import build_agent, query_with_critic

# 1. Suppress Pydantic and Deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. Set environment variable to reduce LlamaIndex's internal logging
os.environ["LLM_LOG_LEVEL"] = "ERROR"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

def main():
    print("==================================================")
    print("🍎 Apple Portfolio Intelligence CLI")
    print("==================================================")
    print("Loading vector database and initializing agent... Please wait.\n")

    try:
        ingestor = PortfolioIngestor()
        index = ingestor.load_index()
        agent, _ = build_agent(index)

    except Exception as e:
        print(f"\n❌ Failed to initialize the engine: {e}")
        sys.exit(1)

    print("\n✅ Agent is ready! (Type 'exit' or 'quit' to stop, 'clear' to reset)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nUser ❯ ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("\nClosing down. Goodbye!")
                break
            if user_input.lower() == 'clear':
                agent.chat_history.clear()
                print("\n[Chat history cleared]")
                continue

            print("\n🍎 Agent is compiling data and running mathematical proofs...")

            try:
                # Call the Critic-in-the-loop function
                final_response = query_with_critic(user_input, agent)
                
                print("=================================================== RESPONSE ===================================================")
                print(f"\n{final_response}")
                print("================================================================================================================")
            
            except Exception as e:
                if "max iterations" in str(e).lower():
                    print("\n⚠️ The analysis was too complex to finish in time. Try asking for a smaller year range.")
                else:
                    print(f"\n⚠️ Error: {e}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()