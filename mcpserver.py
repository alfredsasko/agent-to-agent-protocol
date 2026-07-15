import json
import sys
from pathlib import Path

# Load data relative to this source file so the server works from any cwd.
DATA_PATH = Path(__file__).resolve().parent / "data" / "doctors.json"
doctors: list = json.loads(DATA_PATH.read_text(encoding="utf-8"))


def list_doctors(state: str | None = None, city: str | None = None) -> list[dict]:
    """This tool returns a list of doctors practicing in a specific location. The search is case-insensitive.

    Args:
        state: The two-letter state code (e.g., "CA" for California).
        city: The name of the city or town (e.g., "Boston").

    Returns:
        A JSON string representing a list of doctors matching the criteria.
        If no criteria are provided, an error message is returned.
        Example: '[{"name": "Dr John James", "specialty": "Cardiology", ...}]'
    """
    # Input validation: ensure at least one search term is given.
    if not state and not city:
        return [{"error": "Please provide a state or a city."}]

    target_state = state.strip().lower() if state else None
    target_city = city.strip().lower() if city else None

    return [
        doc
        for doc in doctors
        if (not target_state or doc["address"]["state"].lower() == target_state)
        and (not target_city or doc["address"]["city"].lower() == target_city)
    ]


TOOL_DEFINITION = {
    "name": "list_doctors",
    "description": list_doctors.__doc__,
    "inputSchema": {
        "type": "object",
        "properties": {
            "state": {
                "type": ["string", "null"],
                "description": "Two-letter US state code, such as MA.",
            },
            "city": {
                "type": ["string", "null"],
                "description": "City or town name, such as Boston.",
            },
        },
    },
}


def _write_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _success(request_id: object, result: dict) -> None:
    _write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: object, code: int, message: str) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def run_stdio_server() -> None:
    """Serve the doctor tool over the MCP JSON-RPC stdio transport."""
    for raw_line in sys.stdin:
        request_id: object = None
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}

            if method == "initialize":
                _success(
                    request_id,
                    {
                        "protocolVersion": params.get(
                            "protocolVersion", "2025-06-18"
                        ),
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "doctorserver", "version": "1.0.0"},
                    },
                )
            elif method in {"notifications/initialized", "notifications/cancelled"}:
                continue
            elif method == "ping":
                _success(request_id, {})
            elif method == "tools/list":
                _success(request_id, {"tools": [TOOL_DEFINITION]})
            elif method == "tools/call":
                if params.get("name") != "list_doctors":
                    _error(request_id, -32601, "Unknown tool")
                    continue
                arguments = params.get("arguments") or {}
                result = list_doctors(
                    state=arguments.get("state"),
                    city=arguments.get("city"),
                )
                _success(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result),
                            }
                        ],
                        "isError": False,
                    },
                )
            elif request_id is not None:
                _error(request_id, -32601, f"Unsupported method: {method}")
        except (TypeError, ValueError) as exc:
            _error(request_id, -32602, f"Invalid request: {exc}")


if __name__ == "__main__":
    run_stdio_server()
