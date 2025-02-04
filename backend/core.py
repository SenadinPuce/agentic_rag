from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description_and_args
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

from tools.tools import vector_search_tool, web_search_tool

load_dotenv()


def main(query: str):

    llm = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)

    tools_for_agent = [
        vector_search_tool,
        web_search_tool,
    ]

    system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
    Always try the \"VectorStoreSearch\" tool first. Only use \"WebSearch\" if the vector store does not contain the required information.
    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
    Valid "action" values: "Final Answer" or {tool_names}
    Provide only ONE action per $JSON_BLOB, as shown:"
    ```
    {{
    "action": $TOOL_NAME,
    "action_input": $INPUT
    }}
    ```
    Follow this format:
    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {{
    "action": "Final Answer",
    "action_input": "Final response to human"
    }}
    Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
    Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

    human_prompt = """{input}
    {agent_scratchpad}
    (reminder to always respond in a JSON blob)"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    prompt = prompt.partial(
        tools=render_text_description_and_args(list(tools_for_agent)),
        tool_names=", ".join([t.name for t in tools_for_agent]),
    )

    chain = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=chain,
        tools=tools_for_agent,
        handle_parsing_errors=True,
        verbose=True,
    )

    result = agent_executor.invoke({"input": query})

    print(result)

    return result
