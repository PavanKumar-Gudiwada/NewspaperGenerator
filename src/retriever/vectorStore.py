import os
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from huggingface_hub import login

# Load variables from .env
load_dotenv()

# Hugging Face login
hf_api_key = os.getenv("HF_Token")
if hf_api_key:
    os.environ["HUGGINGFACE_TOKEN"] = hf_api_key
    login(token=hf_api_key)

# Directory for FAISS index
INDEX_DIR = "./faiss_index"


def build_retriever_from_docs(
    documents: list[Document] = None,
    embedding_model_name="google/embeddinggemma-300M",
    chunk_size=500,
    chunk_overlap=50,
    k=5,
):
    """
    Build a FAISS retriever from documents.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, model_kwargs={"device": device}
    )  # performs the tokeinization and generates embeddings for every chunk

    vector_store = None
    rebuild_required = False

    if documents is not None:
        print(
            f"Building new in-memory FAISS index from {len(documents)} provided documents..."
        )
        docs_to_index = documents
        rebuild_required = True  # Force rebuild because it's a new session/document set
    else:
        print("Please provide the documents to refer to")

    if rebuild_required:

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs_to_index)

        # Build FAISS vector store
        vector_store = FAISS.from_documents(split_docs, embedding=hf_embeddings)

    # Ensure a vector store was successfully created/loaded
    if vector_store is None:
        raise RuntimeError("Failed to build the vector store.")

    del hf_embeddings  # Free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory if used

    return vector_store.as_retriever(search_kwargs={"k": k})
