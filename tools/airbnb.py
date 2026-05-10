"""
Airbnb MCP Tool — spawns the @openbnb/mcp-server-airbnb process via stdio
and sends a JSON-RPC tools/call request to search Airbnb listings.
"""
import json
import subprocess
import uuid


def search_airbnb(query: str) -> str:
    """
    Search Airbnb listings using the MCP server.

    Args:
        query: Natural language search query, e.g. 'stays in Goa under 5000'

    Returns:
        Formatted string of Airbnb search results.
    """
    # Build a structured location + query from the natural language input
    request_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": "airbnb_search",
            "arguments": {
                "location": query,
                "adults": 2,
            },
        },
    }

    try:
        proc = subprocess.run(
            ["npx", "-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
            input=json.dumps(request_payload) + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )

        raw = proc.stdout.strip()
        if not raw:
            return f"No results returned from Airbnb MCP. stderr: {proc.stderr[:300]}"

        # The MCP server may return multiple JSON lines; take the last valid one
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
                if "result" in response:
                    content = response["result"].get("content", [])
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts) if texts else str(response["result"])
                if "error" in response:
                    return f"MCP error: {response['error']}"
            except json.JSONDecodeError:
                continue

        return f"Could not parse MCP response: {raw[:500]}"

    except subprocess.TimeoutExpired:
        return "Airbnb MCP request timed out after 30 seconds."
    except FileNotFoundError:
        return "npx not found. Please install Node.js to use the Airbnb MCP tool."
    except Exception as e:
        return f"Airbnb MCP error: {e}"
