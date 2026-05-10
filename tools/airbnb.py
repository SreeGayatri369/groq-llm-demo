"""
Airbnb MCP Tool — safely queries the Airbnb MCP server
and RETURNS A TRUNCATED, LLM-SAFE SUMMARY.
"""
import json
import subprocess
import uuid

MAX_CHARS = 3000  # ✅ HARD safety limit to avoid token explosion


def search_airbnb(query: str) -> str:
    """
    Search Airbnb listings using the MCP server.

    This function ALWAYS returns a TRUNCATED, SAFE response
    so it will NEVER overflow Groq token limits.
    """
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
            return "No Airbnb results found."

        # Take the LAST valid JSON line (MCP behavior)
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue

            try:
                response = json.loads(line)

                if "result" in response:
                    content = response["result"].get("content", [])
                    texts = [
                        c.get("text", "")
                        for c in content
                        if c.get("type") == "text"
                    ]
                    combined = "\n".join(texts)

                    # ✅ HARD TRUNCATION
                    return combined[:MAX_CHARS]

                if "error" in response:
                    return f"Airbnb MCP error: {response['error']}"

            except json.JSONDecodeError:
                continue

        return "Could not parse Airbnb response."

    except subprocess.TimeoutExpired:
        return "Airbnb request timed out."
    except FileNotFoundError:
        return "Node.js (npx) is required to use Airbnb MCP."
    except Exception as e:
        return f"Airbnb tool error: {str(e)}"