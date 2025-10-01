from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def run_qa_query(retriever, query, llm_model_name="gpt-4o-mini", temperature=0):
    """
    Runs a RetrievalQA chain on the given query using the retriever.
    """
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.run(query)
