from dotenv import load_dotenv
import logging
from langgraph.graph import END, StateGraph, START
from models.graph_state import GraphState
from components.edges import decide_to_generate
from components.nodes import (
    retrieve,
    grade_documents,
    generate,
    transform_query,
    web_search,
)

logging.basicConfig(level=logging.INFO)

load_dotenv()

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)


workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)


app = workflow.compile()
