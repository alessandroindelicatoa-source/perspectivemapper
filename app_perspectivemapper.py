"""Entry point kept for the existing Streamlit Community Cloud deployment.

The deployed app was configured with this file as its main module. The application
itself lives in app.py; this file only runs it, so the deployment keeps working
without changing its settings.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
