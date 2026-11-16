"""Streamlit Application Entrypoint for Indonesian Insurance Claim & FNOL Triage.

Forwards to app.streamlit_app.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
