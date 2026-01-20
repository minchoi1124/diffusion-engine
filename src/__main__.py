# src/__main__.py
"""
Package entrypoint for the Diffusion Engine.

Allows execution via:
    python -m src --prompt "..."
"""

from src.main import main

if __name__ == "__main__":
    raise SystemExit(main())