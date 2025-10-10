import os
import torch
from dotenv import load_dotenv
from retriever.dataLoader import load_all_data
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from huggingface_hub import login

# Import hashing service functions
from retriever.hashingService import is_rebuild_required, save_current_hash, INDEX_DIR, DATA_DIR

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
    Build or load a FAISS retriever from documents,
    using a hash to check if rebuild is necessary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, model_kwargs={"device": device}
    ) #performs the tokeinization and generates embeddings for every chunk


    vector_store = None
    rebuild_required = False
    current_data_hash = None # Only used in the default path

    if documents is not None:
        # --- PATH 1: Dynamic In-Memory Index (User-Selected Documents) ---
        print(f"Building new in-memory FAISS index from {len(documents)} provided documents...")
        docs_to_index = documents
        rebuild_required = True # Force rebuild because it's a new session/document set
    else:
        # --- PATH 2: Disk-Based Index (Old Logic/Default) ---

        # --- Hash Check
        rebuild_required, current_data_hash = is_rebuild_required()

        vector_store = None
        
        if not rebuild_required:
            print("Loading existing FAISS index from disk...")
            try:
                vector_store = FAISS.load_local(
                    INDEX_DIR, hf_embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Building index again")
                rebuild_required = True
        
        if rebuild_required:
            print("Building new FAISS index...")
            # Load documents
            docs_to_index = load_all_data("data")

    # --- Index Creation Logic (Shared by both paths) ---
    if rebuild_required:
        if not docs_to_index:
            raise RuntimeError("Failed to load any documents for indexing.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs_to_index)

        # Build FAISS vector store
        vector_store = FAISS.from_documents(split_docs, embedding=hf_embeddings)

        # Only save the index and hash if we are in the default disk-based path
        if documents is None:
            # Save index
            vector_store.save_local(INDEX_DIR)
            # Save the new hash after a successful build
            save_current_hash(current_data_hash)
            print(f"FAISS index saved locally at {INDEX_DIR}")
        else:
            print("In-memory FAISS index built successfully (not saved to disk).")


    # Ensure a vector store was successfully created/loaded
    if vector_store is None:
        raise RuntimeError("Failed to build or load the vector store.")
    
    del hf_embeddings  # Free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory if used

    return vector_store.as_retriever(search_kwargs={"k": k})