
from ..core.config import LANGSMITH_ENABLED, LANGSMITH_API_KEY
from langsmith import Client, traceable

def get_langsmith_client():
    if LANGSMITH_ENABLED and LANGSMITH_API_KEY:
        return Client(api_key=LANGSMITH_API_KEY)
    return None

def trace_function(fn):
    return traceable(fn)
