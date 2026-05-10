# Groq LLM Demo with LangGraph Agent

AI assistant powered by **Groq LLM** + **LangGraph** with tool calling for Airbnb search, weather, and web search.

## Features

- 🤖 **LangGraph Agent** with tool routing and state tracking
- 🏠 **Airbnb MCP** integration via stdio subprocess
- 🌤 **Weather API** for current weather data
- 🔍 **Tavily Search** for web queries
- 📊 **LangSmith Tracing** for debugging agent flows
- 🎨 **Beautiful UI** with tool usage visualization

## Quick Start (GitHub Codespaces)

1. Click **Code → Codespaces → Create codespace**
2. Wait for container to build
3. Run:
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Open forwarded port 8000 in browser

## Local Setup

1. Clone the repo
2. Create `.env` file:
   ```
   GROQ_API_KEY=your_key
   WEATHER_API_KEY=your_key
   TAVILY_API_KEY=your_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key
   LANGCHAIN_PROJECT=groq-llm-demo
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

## Project Structure

```
groq-llm-demo/
├── agent/
│   ├── graph.py          # LangGraph agent with tool routing
│   └── __init__.py
├── llm/
│   ├── groq_client.py    # Groq LLM wrapper
│   └── __init__.py
├── tools/
│   ├── airbnb.py         # Airbnb MCP tool
│   ├── weather.py        # Weather API tool
│   ├── web_search.py     # Tavily search tool
│   └── __init__.py
├── templates/
│   └── index.html        # Chat UI
├── app.py                # FastAPI server
├── config.py             # Environment config
├── requirements.txt
└── README.md
```

## Example Queries

- "Find Airbnb stays in Goa under ₹5000"
- "What's the weather in Mumbai?"
- "Who is the prime minister of India?"
- "Compare Airbnbs in Delhi vs Bangalore"

## LangSmith Tracing

View agent execution traces at [smith.langchain.com](https://smith.langchain.com) — see tool selection, state transitions, and LLM calls in real-time.

## API Keys

- **Groq**: [console.groq.com](https://console.groq.com)
- **Weather**: [weatherapi.com](https://www.weatherapi.com) (free)
- **Tavily**: [tavily.com](https://tavily.com) (free)
- **LangSmith**: [smith.langchain.com](https://smith.langchain.com) (free)

## Tech Stack

- **LLM**: Groq (llama-3.3-70b-versatile)
- **Framework**: LangGraph + LangChain
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML + Tailwind CSS
- **MCP**: @openbnb/mcp-server-airbnb (Node.js)
