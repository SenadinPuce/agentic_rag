from langchain_core.documents import Document
import logging
from chains.question_rewriter_chain import question_rewriter_chain
from tools.agent_tools import vector_search_tool, web_search_tool
from chains.grade_documents_chain import grade_documents_chain
from chains.rag_chain import rag_chain


def retrieve(state):

    logging.info("---RETRIEVE---")
    question = state["question"]

    documents = vector_search_tool.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    logging.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain(documents, question)
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):

    logging.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = grade_documents_chain(d["content"], question)
        grade = score.binary_score
        if grade == "yes":
            logging.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            logging.info("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    if len(filtered_docs) / len(documents) <= 0.7:
        web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):

    logging.info("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    improved_question = question_rewriter_chain(question)
    return {"documents": documents, "question": improved_question}


def web_search(state):

    logging.info("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke(question)
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}
