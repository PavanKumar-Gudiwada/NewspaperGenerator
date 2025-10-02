import os
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from huggingface_hub import login

# Load variables from .env
load_dotenv()

# Get the Hugging Face API key
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_TOKEN"] = hf_api_key

login()

# Define directory to save/load FAISS index files
INDEX_DIR = "./faiss_index"

def build_retriever_from_texts(texts, embedding_model_name="google/embeddinggemma-300M", chunk_size=500, chunk_overlap=50, k=5):
    """
    Split texts, create embeddings, and return a LangChain retriever.
    """
    # Load saved FAISS index if exists
    if os.path.exists(INDEX_DIR):
        print("Loading existing FAISS index from disk...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})
        vector_store = FAISS.load_local(INDEX_DIR, hf_embeddings, allow_dangerous_deserialization=True)
    else:
        print("Building new FAISS index...")
        # Split texts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(" ".join(texts))

        # Create embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

        # Build FAISS vector store
        vector_store = FAISS.from_texts(chunks, embedding=hf_embeddings)

        # Save the index for future use
        vector_store.save_local(INDEX_DIR)
        print(f"FAISS index saved locally at {INDEX_DIR}")

    # Return retriever
    return vector_store.as_retriever(search_kwargs={"k": k})
