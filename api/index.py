import sys
import os

# Append project root to sys.path for Vercel Python runtime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app

