from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def question_rewriter_chain(question: str) -> str:
    system = """You are a question re-writer that converts an input question to a better version that is optimized \\n 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \\n\\n {question} \\n Formulate an improved question.",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)

    question_rewriter = prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": question})
