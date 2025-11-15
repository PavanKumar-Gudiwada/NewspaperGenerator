from pipeline.rag_llm_pipeline import rag_llm_pipeline
import torch

from generator.parseOutput import parse_llm_json

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Enter you query:")
    user_query = input()

    answer = rag_llm_pipeline(user_query)

    structured_output = parse_llm_json(answer["result"])

    print("Title:", structured_output.get("title", "No title found."))
    print("Article:", structured_output.get("article", "No article found."))
