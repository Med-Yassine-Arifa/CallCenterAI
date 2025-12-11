"""
Pytest configuration file for all tests.
Configures Python path to include src directory.
"""
import sys
from pathlib import Path

# Add the src directory to Python path for all tests
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
