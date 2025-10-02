from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from generator.tools import get_llm

def run_qa_query(retriever, query, llm_model_name=None, temperature=0):
    """
    Runs a RetrievalQA chain on the given query using the retriever.
    """
    # Use LLM from tools.py
    llm = get_llm(model_name=llm_model_name, temperature=temperature)

    # Define a custom prompt template
    PROMPT_TEMPLATE = """
    Answer Using only the following context:
    {context}
    Write a newspaper article of 3 paragraphs using the above context with an interesting title using the topic from the question.
    Question: {question}
    Provide a detailed answer.
    Don’t justify your answers.
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    # RetrievalQA chain combines an information retrieval system with a language model to answer questions using external data, providing factual answers.
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                           retriever=retriever, chain_type_kwargs={"prompt": prompt_template})
    return qa_chain.invoke(query)
