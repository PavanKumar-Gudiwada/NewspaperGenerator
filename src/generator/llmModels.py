import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint

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
    
    elif provider == "huggingface":
        # Use Hugging Face Hub (requires HF_TOKEN)
        model_id = model_name or os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        hf_token = os.getenv("HF_Token")

        if not hf_token:
            raise ValueError("Missing Hugging Face API token. Please set HF_Token env var.")

        return HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        )

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
