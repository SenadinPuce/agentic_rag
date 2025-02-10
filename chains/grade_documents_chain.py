from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from models.grade_documents import GradeDocuments


def grade_documents_chain(document: str, question: str) -> GradeDocuments:
    llm = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    grade = retrieval_grader.invoke({"document": document, "question": question})

    return grade
