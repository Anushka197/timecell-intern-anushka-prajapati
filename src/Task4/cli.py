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
from engine import build_agent

# Suppress debug/info logging from external libraries to keep the CLI clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

## To clean warnings


# 1. Suppress Pydantic and Deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. Set environment variable to reduce LlamaIndex's internal logging
os.environ["LLM_LOG_LEVEL"] = "ERROR"


def main():
    print("==================================================")
    print("🍎 Apple Portfolio Intelligence CLI")
    print("==================================================")
    print("Loading vector database and initializing agent... Please wait.\n")

    try:
        # 1. Load the vector index
        ingestor = PortfolioIngestor()
        index = ingestor.load_index()

        # 2. Build the ReAct agent
        # We unpack the tuple (agent, debug_handler) returned by build_agent
        agent, _ = build_agent(index)

    except Exception as e:
        print(f"\n❌ Failed to initialize the engine: {e}")
        sys.exit(1)

    print("\n✅ Agent is ready! (Type 'exit' or 'quit' to stop, 'clear' to reset)")
    print("-" * 50)

    # 3. Interactive Chat Loop
    while True:
        try:
            # Get user input
            user_input = input("\nUser ❯ ").strip()
            
            # Handle empty input
            if not user_input:
                continue
                
            # Handle exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nClosing down. Goodbye!")
                break
                
            # Handle reset chat history
            if user_input.lower() == 'clear':
                agent.chat_history.clear()
                print("\n[Chat history cleared]")
                continue

            print("\nAgent is thinking... (this may take a few moments)\n")
            
            # Execute the query
            # Inside your while loop in cli.py
            print("\n🍎 Agent is analyzing Apple's filings...")

            try:
                response = agent.chat(user_input)
                print("=================================================== RESPONSE ===================================================")
                print(f"\n{response}")
                print("================================================================================================================")
            except Exception as e:
                if "max iterations" in str(e).lower():
                    print("\n⚠️ The analysis was too complex to finish in time. Try asking for a smaller year range.")
                else:
                    print(f"\n⚠️ Error: {e}")
            
            # Print the final response clearly
            # print(str(response))
            # print("================================================================================================================")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred during the query: {e}")

if __name__ == "__main__":
    main()