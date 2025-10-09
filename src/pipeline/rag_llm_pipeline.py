from retriever.vectorStore import build_retriever_from_docs
from generator.llmQA import run_qa_query
from langchain_core.documents import Document # Import Document type for clarity
from pipeline.pipelineHelper import format_rag_response

def rag_llm_pipeline(data_folder="data", query="What is the main topic of the documents?", documents: list[Document] = None):

    # 1. Build retriever
    retriever = build_retriever_from_docs(documents=documents)

    # 2. Generator: LLM answer using retrieved documents
    answer = run_qa_query(retriever, query, temperature=0.3)
    
    return answer