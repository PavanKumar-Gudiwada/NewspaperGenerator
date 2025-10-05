import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_llm(model_name=None, temperature=0):
    """
    Returns an LLM client depending on the LLM_PROVIDER environment variable.
    Supported: openai, ollama
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return ChatOpenAI(
            model_name=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature
        )

    elif provider == "ollama":
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "llama2"),
            temperature=temperature
        )

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
