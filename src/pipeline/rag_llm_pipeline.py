from retriever.dataLoader import load_all_data
from retriever.vectorStore import build_retriever_from_texts
from generator.llmQA import run_qa_query

def rag_llm_pipeline(data_folder="data", query="What is the main topic of the documents?"):
    # 1. Load documents
    all_texts = load_all_data("data")

    # 2. Build retriever
    retriever = build_retriever_from_texts(all_texts)

    # 3. Run query
    answer = run_qa_query(retriever, query)
    print("Answer:", answer)