import os
import json

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import END

from src.rag import search_meetings


load_dotenv()


# ============================================================
# TOOLS
# ============================================================

@tool
def search_meeting_memory(question: str) -> str:
    """
    Search previous meeting transcripts using semantic search.
    Use this for questions about meeting discussions,
    decisions, or historical information.
    """

    results = search_meetings(question, k=3)

    if not results:
        return "No relevant meeting information was found."

    context = []

    for result in results:
        context.append(result.page_content)

    return "\n\n".join(context)


@tool
def get_action_items() -> str:
    """
    Get action items from the current meeting.
    """

    path = "data/current_meeting.json"

    if not os.path.exists(path):
        return "No meeting analysis is available."

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("action_items", [])

    if not items:
        return "No action items were identified."

    output = []

    for item in items:
        output.append(
            f"Task: {item['task']}\n"
            f"Owner: {item['owner']}\n"
            f"Deadline: {item['deadline']}"
        )

    return "\n\n".join(output)


@tool
def get_meeting_risks() -> str:
    """
    Get risks and unresolved issues from the current meeting.
    """

    path = "data/current_meeting.json"

    if not os.path.exists(path):
        return "No meeting analysis is available."

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    risks = data.get("risks", [])

    if not risks:
        return "No major risks were identified."

    return "\n".join(
        f"- {risk}"
        for risk in risks
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


tools = [
    search_meeting_memory,
    get_action_items,
    get_meeting_risks
]


llm_with_tools = llm.bind_tools(tools)


# ============================================================
# AGENT NODE
# ============================================================

SYSTEM_PROMPT = """
You are MeetingMind, an intelligent meeting assistant.

You have access to three tools:

1. search_meeting_memory
   - Use for questions about meeting discussions,
     decisions, and information from the meeting.

2. get_action_items
   - Use when the user asks about tasks,
     owners, responsibilities, or deadlines.

3. get_meeting_risks
   - Use when the user asks about risks,
     concerns, or unresolved issues.

Choose the appropriate tool based on the user's question.

Do not invent information.

After receiving tool results, provide a concise,
clear answer to the user.

If the information cannot be found, say so clearly.
"""


def agent_node(state: MessagesState):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# ROUTING
# ============================================================

def should_continue(state: MessagesState):

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls"):

        if last_message.tool_calls:
            return "tools"

    return END


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(MessagesState)

builder.add_node(
    "agent",
    agent_node
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "agent"
)

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

builder.add_edge(
    "tools",
    "agent"
)


meeting_agent = builder.compile()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def ask_meeting_agent(question: str):

    result = meeting_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    return result["messages"][-1].content