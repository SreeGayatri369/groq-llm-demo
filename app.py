"""
FastAPI app — serves the chat UI and exposes /chat endpoint.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import run_agent

app = FastAPI(title="Groq AI Agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    tool_used: str
    tool_output: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message)
    return ChatResponse(**result)


@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
