"""
Streamlit entry point for local dev and Streamlit Cloud deployment.

Streamlit Cloud auto-detects this file at the repo root.
It adds src/ to the Python path, then re-exports the real app.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Re-export the real app module — Streamlit reads its top-level code
from study_agent.app import *
