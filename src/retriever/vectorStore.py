import os
import torch
from dotenv import load_dotenv
from retriever.dataLoader import load_all_data
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login

# Load variables from .env
load_dotenv()

# Hugging Face login
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if hf_api_key:
    os.environ["HUGGINGFACE_TOKEN"] = hf_api_key
    login()

# Directory for FAISS index
INDEX_DIR = "./faiss_index"


def build_retriever_from_docs(
    embedding_model_name="google/embeddinggemma-300M",
    chunk_size=500,
    chunk_overlap=50,
    k=5,
):
    """
    Build or load a FAISS retriever from documents.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, model_kwargs={"device": device}
    )

    if os.path.exists(INDEX_DIR):
        print("Loading existing FAISS index from disk...")
        vector_store = FAISS.load_local(
            INDEX_DIR, hf_embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Building new FAISS index...")
        # Load documents
        docs = load_all_data("data")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)

        # Build FAISS vector store
        vector_store = FAISS.from_documents(split_docs, embedding=hf_embeddings)

        # Save index
        vector_store.save_local(INDEX_DIR)
        print(f"FAISS index saved locally at {INDEX_DIR}")

    return vector_store.as_retriever(search_kwargs={"k": k})
