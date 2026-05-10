"""
LangGraph Agent — routes user queries to the correct tool and returns
a structured response with tool name, tool output, and final answer.

Flow: User Input → Agent Node → Tool Decision → Tool Node → Final Response
"""
import json
import logging
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from config import GROQ_API_KEY
from tools.airbnb import search_airbnb
from tools.weather import get_weather
from tools.web_search import web_search

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── LangChain tool wrappers ──────────────────────────────────────────────────

@tool
def airbnb_tool(query: str) -> str:
    """Search Airbnb vacation rentals, apartments, homes, and stays. Use ONLY for Airbnb-specific queries or when user asks about vacation rentals, short-term stays, apartments, or accommodation bookings. Examples: 'Find Airbnb in Goa', 'vacation rentals in Mumbai', 'apartments to rent in Bangalore'."""
    logger.info("Tool called: airbnb | query: %s", query)
    return search_airbnb(query)


@tool
def weather_tool(location: str) -> str:
    """Get current weather for a city or location. Use for weather, temperature, climate queries."""
    logger.info("Tool called: weather | location: %s", location)
    return get_weather(location)


@tool
def web_search_tool(query: str) -> str:
    """Search the web for general information, news, facts, hotels, restaurants, tourist attractions, or anything not related to Airbnb vacation rentals or weather. Use for queries about hotels, top places, general recommendations."""
    logger.info("Tool called: web_search | query: %s", query)
    return web_search(query)


TOOLS = [airbnb_tool, weather_tool, web_search_tool]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── Graph state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_used: str
    tool_output: str


# ── Graph nodes ──────────────────────────────────────────────────────────────

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0,
    http_client=httpx.Client(verify=False, follow_redirects=True),
).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Call the LLM; it decides whether to use a tool or answer directly."""
    logger.info("Agent node: processing %d messages", len(state["messages"]))
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute whichever tool the LLM chose."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])

    if not tool_calls:
        return {"tool_used": "none", "tool_output": ""}

    call = tool_calls[0]
    name = call["name"]
    args = call["args"]

    logger.info("Executing tool: %s | args: %s", name, args)

    tool_fn = TOOL_MAP.get(name)
    if not tool_fn:
        output = f"Unknown tool: {name}"
    else:
        try:
            output = tool_fn.invoke(args)
        except Exception as e:
            output = f"Tool error: {e}"

    tool_msg = ToolMessage(content=output, tool_call_id=call["id"])
    return {
        "messages": [tool_msg],
        "tool_used": name,
        "tool_output": output,
    }


def should_use_tool(state: AgentState) -> str:
    """Router: go to tool_node if LLM requested a tool, else END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", []):
        return "tool"
    return "end"


# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_use_tool, {"tool": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_graph = build_graph()


# ── Public API ───────────────────────────────────────────────────────────────

def run_agent(user_input: str) -> dict:
    """
    Run the LangGraph agent for a user query.

    Returns:
        {
            "answer": str,
            "tool_used": str,   # "none" | "airbnb_tool" | "weather_tool" | "web_search_tool"
            "tool_output": str,
        }
    """
    logger.info("=== Agent run start | input: %s ===", user_input)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_used": "none",
        "tool_output": "",
    }

    final_state = _graph.invoke(initial_state)

    # Last AI message is the final answer
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
