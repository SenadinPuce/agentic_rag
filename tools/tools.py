import os

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langsmith import traceable

@tool
@traceable
def web_search_tool(query: str) -> str:
    """
    Use this to perform a web search for information.
    """

    tavily_tool = TavilySearchResults()

    results = tavily_tool.invoke({"query": query})

    formatted_results = []

    for result in results:
        formatted_results.append(
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {result.get('content', 'N/A')}\n"
        )

    return formatted_results


@tool("VectorStoreSearch")
@traceable
def vector_search_tool(query: str) -> str:
    """
    Use this to search the vector store for information.
    """

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    results = vector_store.similarity_search(query)

    formatted_result = []

    for doc in results:
        result_entry = {"content": doc.page_content, "metadata": doc.metadata}
        formatted_result.append(result_entry)

    return formatted_result