from retriever.vectorStore import build_retriever_from_docs
from generator.llmQA import run_qa_query

def rag_llm_pipeline(data_folder="data", query="What is the main topic of the documents?"):

    # 1. Build retriever
    retriever = build_retriever_from_docs()

    # 2. Generator: LLM answer using retrieved documents
    answer = run_qa_query(retriever, query, temperature=0.3)
    
    return answer