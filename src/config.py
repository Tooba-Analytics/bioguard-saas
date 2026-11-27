import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = "openai/gpt-oss-120b"
API_TIMEOUT_SEC = 8
MAX_RETRIES = 2
MAX_WORKERS = 8
MAX_CRITIC_RETRIES = 1