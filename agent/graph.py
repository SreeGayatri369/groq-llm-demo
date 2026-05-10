"""
LangGraph Agent — SAFE token flow, correct tool routing,
and FINAL answer generation after tool execution.
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

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── System rules ──────────────────────────────────────────────────────────

SYSTEM_RULE = SystemMessage(
    content=(
        "You are an AI assistant.\n\n"
        "RULES:\n"
        "- Any query about hotels, stays, accommodation, rooms, rentals, or budget stays "
        "MUST use the Airbnb tool.\n"
        "- Use web search only for general informational queries.\n"
        "- Use weather tool only for weather-related queries.\n"
        "- After using a tool, always explain the result clearly to the user.\n"
        "- Never call more than one tool per user question.\n"
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
    """General info only (NOT hotels or stays)."""
    logger.info("Tool called: web_search | query: %s", query)
    return web_search(query)


TOOLS = [airbnb_tool, weather_tool, web_search_tool]
TOOL_MAP = {t.name: t for t in TOOLS}

# ── State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_used: str
    tool_output: str

# ── LLM ───────────────────────────────────────────────────────────────────

llm = (
    ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
        http_client=httpx.Client(verify=False, follow_redirects=True),
    )
    .bind_tools(TOOLS)
)

# ── Nodes ────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    Agent node: decides whether to call a tool
    OR generates the final human-readable answer.
    """
    logger.info("Agent node: processing %d messages", len(state["messages"]))

    # Keep context small to avoid token overflow
    messages = [SYSTEM_RULE] + state["messages"][-4:]

    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """
    Executes the selected tool and returns the output
    to the agent for final explanation.
    """
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
        output = tool_fn.invoke(args) if tool_fn else "Unknown tool."
    except Exception as e:
        output = f"Tool error: {e}"

    # Safety cap to prevent token explosion
    output = output[:3000]

    tool_msg = ToolMessage(
        content=output,
        tool_call_id=call["id"],
    )

    return {
        "messages": [tool_msg],
        "tool_used": name,
        "tool_output": output,
    }

# ✅ MINIMAL FIX IS HERE
# This ensures the agent ALWAYS runs once after tool execution

def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]

    # If LLM requested a tool → go to tool
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", []):
        return "tool"

    # Otherwise → end (final answer already generated)
    return "end"

# ── Graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            "tool": "tools",
            "end": END,
        },
    )

    # ✅ CRITICAL: tool ALWAYS returns to agent for final explanation
    graph.add_edge("tools", "agent")

    return graph.compile()

_graph = build_graph()

# ── Public API ────────────────────────────────────────────────────────────

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