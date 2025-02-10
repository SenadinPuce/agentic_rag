from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def rag_chain(context: str, question: str) -> str:
    template = """"
    You are a helpful assistant that answers questions based on the following context.'
    Use the provided context to answer the question.
    Context: {context}
    Question: {question}
    Answer:

    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)

    rag = prompt | llm | StrOutputParser()
    return rag.invoke({"context": context, "question": question})
