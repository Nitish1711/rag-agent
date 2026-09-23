from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

from app.config import LLM_MODEL
from app.prompts import SYSTEM_PROMPT
from app.state import AgentState
from app.tools import retrieve_document


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)

tools = [retrieve_document]

llm_with_tools = llm.bind_tools(tools)

tools_dict = {
    tool.name: tool
    for tool in tools
}


def call_llm(state: AgentState):

    messages = state["messages"]

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }

    response = llm_with_tools.invoke(
        [system_message] + list(messages)
    )

    return {
        "messages": [response]
    }


def take_action(state: AgentState):

    last_message = state["messages"][-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        []
    )

    tool_messages = []

    for tool_call in tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        tool_call_id = tool_call["id"]

        if tool_name not in tools_dict:

            result = (
                f"Unknown tool requested: {tool_name}"
            )

        else:

            try:

                result = tools_dict[
                    tool_name
                ].invoke(tool_args)

            except Exception as exc:

                result = (
                    f"Tool execution failed: {str(exc)}"
                )

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id
            )
        )

    return {
        "messages": tool_messages
    }


def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "tool_calls",
        None
    ):
        return "retriever_agent"

    return END


workflow = StateGraph(AgentState)

workflow.add_node(
    "llm",
    call_llm
)

workflow.add_node(
    "retriever_agent",
    take_action
)

workflow.set_entry_point("llm")

workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "retriever_agent": "retriever_agent",
        END: END,
    }
)

workflow.add_edge(
    "retriever_agent",
    "llm"
)

rag_agent = workflow.compile()