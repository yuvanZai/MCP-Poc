# math_mcp_server.py
# This file defines an MCP server named "Math" with prompts, resources, and tools
# for basic mathematical operations.
import json
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server with the name "Math"
mcp = FastMCP("Math")

# --- Prompts ---
@mcp.prompt()
def example_prompt(question: str) -> str:
    """
    Example prompt description.
    This prompt provides instructions for a math assistant.
    """
    return f"""
    You are a math assistant. Answer the question.
    Question: {question}
    """

@mcp.prompt()
def system_prompt() -> str:
    """
    System prompt description.
    This prompt provides general instructions for an AI assistant.
    """
    return """
    You are an AI assistant use the tools
if needed.
    """

# --- Resources ---
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    Get a personalized greeting.
    This is a dynamic resource that generates a greeting based on the provided name.
    """
    return f"Hello, {name}!"

@mcp.resource("config://app")
def get_config() -> str:
    """
    Static configuration data.
    This is a static resource providing application configuration.
    """
    return "App configuration here"

# --- Tools ---
@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers.
    This tool performs addition of two integer numbers.
    """
    print(f"MCP Server: Received call to 'add' with a={a}, b={b}")
    result = a + b
    print(f"MCP Server: 'add' returning result: {result}")
    return result

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    This tool performs multiplication of two integer numbers.
    """
    print(f"MCP Server: Received call to 'multiply' with a={a}, b={b}")
    result = a * b
    print(f"MCP Server: 'multiply' returning result: {result}")
    return result

# Main execution block: runs the server.
if __name__ == "__main__":
    print("Math MCP Server: Starting...")
    mcp.run()
    print("Math MCP Server: Shutting down.")