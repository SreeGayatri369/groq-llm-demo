"""
LangGraph Agent — SAFE token flow, correct tool routing,
and no infinite loops.
"""
import logging
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from config import GROQ_API_KEY
from tools.airbnb import search_airbnb
from tools.weather import get_weather
from tools.web_search import web_search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ✅ SYSTEM RULE: Hotels / stays MUST go to Airbnb
SYSTEM_RULE = SystemMessage(
    content=(
        "You are an AI assistant.\n"
        "IMPORTANT RULE:\n"
        "- Any query about hotels, stays, accommodation, rooms, rentals, budget stays\n"
        "  MUST use the Airbnb tool.\n"
        "- Use web search ONLY for non-booking information.\n"
        "- Never call more than ONE tool per user question.\n"
    )
)


# ── Tools ────────────────────────────────────────────────────────────────

@tool
def airbnb_tool(query: str) -> str:
    """Use for ALL hotel, stay, accommodation, rental, or budget stay queries."""
    logger.info("Tool called: airbnb | query: %s", query)
    return search_airbnb(query)


@tool
def weather_tool(location: str) -> str:
    """Weather queries only."""
    logger.info("Tool called: weather | location: %s", location)
    return get_weather(location)


@tool
def web_search_tool(query: str) -> str:
    """General info ONLY — NOT hotels or stays."""
    logger.info("Tool called: web_search | query: %s", query)
    return web_search(query)


TOOLS = [airbnb_tool, weather_tool, web_search_tool]
TOOL_MAP = {t.name: t for t in TOOLS}

# ── State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_used: str
    tool_output: str


# ── LLM ─────────────────────────────────────────────────────────────────

llm = (
    ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
        http_client=httpx.Client(verify=False, follow_redirects=True),
    )
    .bind_tools(TOOLS)
)


# ── Nodes ───────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    logger.info("Agent node: processing %d messages", len(state["messages"]))

    # ✅ KEEP CONTEXT SMALL (last 4 messages only)
    messages = [SYSTEM_RULE] + state["messages"][-4:]

    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])

    if not tool_calls:
        return {"tool_used": "none", "tool_output": ""}

    call = tool_calls[0]
    name = call["name"]
    args = call["args"]

    logger.info("Executing tool: %s | args: %s", name, args)

    tool_fn = TOOL_MAP.get(name)

    try:
        output = tool_fn.invoke(args) if tool_fn else "Unknown tool"
    except Exception as e:
        output = f"Tool failure: {e}"

    # ✅ Tool output NEVER goes back to tool discovery
    tool_msg = ToolMessage(
        content=output[:3000],  # ✅ FINAL safety guard
        tool_call_id=call["id"]
    )

    return {
        "messages": [tool_msg],
        "tool_used": name,
        "tool_output": output[:3000],
    }


def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]

    # ✅ Tool already used → END
    if state.get("tool_used") not in ("none", "", None):
        return "end"

    if isinstance(last, AIMessage) and getattr(last, "tool_calls", []):
        return "tool"

    return "end"


# ── Graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_use_tool, {
        "tool": "tools",
        "end": END
    })
    graph.add_edge("tools", "agent")
    return graph.compile()


_graph = build_graph()


def run_agent(user_input: str) -> dict:
    logger.info("=== Agent run start | input: %s ===", user_input)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_used": "none",
        "tool_output": "",
    }

    final_state = _graph.invoke(initial_state)

    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    result = {
        "answer": answer,
        "tool_used": final_state.get("tool_used", "none"),
        "tool_output": final_state.get("tool_output", ""),
    }

    logger.info("=== Agent run end | tool: %s ===", result["tool_used"])
    return result