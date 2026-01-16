import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ========== CRITICAL: DISABLE LANGSMITH ==========
# This must be done EARLY to avoid logging overhead
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# ================================================

# Database URL for SQLAlchemy / LangChain
# Default: SQLite file 'datawarehouse.db'
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///datawarehouse.db")

# Ollama configuration - OPTIMIZED FOR SPEED
OLLAMA_CONFIG = {
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),  # Use smaller/faster model if available
    "temperature": 0.1,  #  Fixed to 0.1 for deterministic, fast responses
    "timeout": 120,  #  Reduced from 600 to 120 seconds
}