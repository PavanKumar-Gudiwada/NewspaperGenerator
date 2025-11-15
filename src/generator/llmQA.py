from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from generator.llmModels import get_llm


def run_qa_query(retriever, query, llm_model_name=None, temperature=0):
    """
    Runs a retrieval → prompt → LLM → JSON parsing pipeline using LCEL.
    """
    llm = get_llm(model_name=llm_model_name, temperature=temperature)

    # Function to join retrieved docs into a single context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    PROMPT_TEMPLATE = """
        Answer using only the following context:
        {context}

        If the context does not contain relevant information, say "I don’t know."

        Write a newspaper article of 3 paragraphs using the above context with an interesting title using the topic from the question.

        Question: {question}

        Provide a detailed answer.
        Don’t justify your answers.
        Don’t give information not mentioned in the CONTEXT INFORMATION.
        Do not say "according to the context" or "mentioned in the context" or similar.

        Return your answer in **JSON** with this format:
        {{
            "title": "...",
            "article": "..."
        }}
        """

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    parser = JsonOutputParser()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )

    response = chain.invoke(query)

    return chain.invoke(query)
