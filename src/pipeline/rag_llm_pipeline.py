from retriever.vectorStore import build_retriever_from_texts
from generator.llmQA import run_qa_query

def rag_llm_pipeline(data_folder="data", query="What is the main topic of the documents?"):

    # 1. Build retriever
    retriever = build_retriever_from_texts()

    # 2. Run query
    answer = run_qa_query(retriever, query)
    print("Answer:", answer)