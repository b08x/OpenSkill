import os
from pathlib import Path
from dotenv import dotenv_values
from typing import Optional

def get_openrouter_key() -> str:
    """
    Retrieves the OPENROUTER_API_KEY exclusively from .env.local.
    Does NOT fallback to system environment variables.
    """
    # Determine the project root (where .env.local should reside)
    # Strategy: Search upwards from this file's location to find the workspace root
    # then check for .env.local there.
    
    current = Path(__file__).resolve()
    # We expect to be in OpenSkillLib/src/openskill/utils/config.py
    # or similar structure. We search for .env.local in parents.
    for parent in current.parents:
        env_local = parent / ".env.local"
        if env_local.exists():
            config = dotenv_values(env_local)
            return config.get("OPENROUTER_API_KEY", "")

    # Final fallback to CWD
    env_local = Path.cwd() / ".env.local"
    if env_local.exists():
        config = dotenv_values(env_local)
        return config.get("OPENROUTER_API_KEY", "")
    
    return ""
