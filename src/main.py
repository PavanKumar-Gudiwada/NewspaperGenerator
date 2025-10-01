from src.pipeline.rag_llm_pipeline import rag_llm_pipeline
import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Enter you query:")
    user_query = input()
    rag_llm_pipeline("data", user_query)