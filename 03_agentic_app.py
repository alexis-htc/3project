"""
03_agentic_app.py
-----------------
Entry point for UDA-Hub - Universal Decision Agent.

How to run:
    1. Set up databases first:
       python 01_external_db_setup.py
       python 02_core_db_setup.py

    2. Create .env with your OpenAI API key:
       cp .env.example .env
       # Edit .env and set OPENAI_API_KEY

    3. Run the agent:
       python 03_agentic_app.py
"""

import os
import sys

# Ensure the solution directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import chat_interface


if __name__ == "__main__":
    chat_interface()
