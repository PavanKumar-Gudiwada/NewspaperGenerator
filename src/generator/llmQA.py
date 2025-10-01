from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def run_qa_query(retriever, query, llm_model_name="text-davinci-003", temperature=0):
    """
    Runs a RetrievalQA chain on the given query using the retriever.
    """
    llm = OpenAI(model_name=llm_model_name, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.run(query)
