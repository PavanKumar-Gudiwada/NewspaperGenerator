from pipeline.rag_llm_pipeline import rag_llm_pipeline
import torch
# Import the original loader to simulate loading the user-selected documents
from retriever.dataLoader import load_all_data

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Enter you query:")
    user_query = input()
    
    # --- Example Simulation: Load documents dynamically (Path 1) ---
    # To test the new path, uncomment the following lines:
    # print("Simulating user selecting ALL documents from 'data' folder...")
    # user_selected_docs = load_all_data("data")
    # answer = rag_llm_pipeline("data", user_query, documents=user_selected_docs)

    # --- Example Simulation: Use the default disk/hashing path (Path 2) ---
    # To test the old path, use the original call:
    answer = rag_llm_pipeline("data", user_query)

    print("Answer:", answer)