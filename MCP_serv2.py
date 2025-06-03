# mcp_server.py
"""
Fixed MCP Server with proper initialization options
"""
import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("demo-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="calculate",
            description="Perform basic math calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2', '10 * 3')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="count_words",
            description="Count words in a text",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to count words in"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="reverse_text",
            description="Reverse the order of characters in text",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to reverse"
                    }
                },
                "required": ["text"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "calculate":
        try:
            expression = arguments["expression"]
            # Simple and safe evaluation for basic math
            allowed_chars = set("0123456789+-*/.() ")
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return [TextContent(type="text", text=f"Result: {result}")]
            else:
                return [TextContent(type="text", text="Error: Invalid characters in expression")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "count_words":
        text = arguments["text"]
        word_count = len(text.split())
        return [TextContent(type="text", text=f"Word count: {word_count}")]
    
    elif name == "reverse_text":
        text = arguments["text"]
        reversed_text = text[::-1]
        return [TextContent(type="text", text=f"Reversed: {reversed_text}")]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    # Option 1: Try with initialization options
    try:
        async with stdio_server() as streams:
            await app.run(*streams, {})  # Empty initialization options
    except TypeError:
        # Option 2: If that fails, try the older API
        try:
            async with stdio_server() as streams:
                await app.run(*streams)
        except Exception as e:
            print(f"Error running server: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)