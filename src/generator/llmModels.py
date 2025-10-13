import os
import torch
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login

def get_llm(model_name=None, temperature=0):
    """
    Returns an LLM client depending on the LLM_PROVIDER environment variable.
    Supported: openai, ollama
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return ChatOpenAI(
            model_name=model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            temperature=temperature,
            timeout=60,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif provider == "ollama":
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "llama2"),
            temperature=temperature
        )
    
    elif provider == "huggingface":
        # Hugging Face login
        hf_api_key = os.getenv("HF_Token")
        if hf_api_key:
            os.environ["HUGGINGFACE_TOKEN"] = hf_api_key
            login(token=hf_api_key)

        hf_model_name = model_name or os.getenv("HF_MODEL", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            # Set generation parameters
            max_new_tokens=512,  # Max length for the generated article
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        return HuggingFacePipeline(pipeline=pipe)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
