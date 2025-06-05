# file_system_mcp_server.py
# This file defines an MCP server named "FileSystem" with tools for local file operations.

from mcp.server.fastmcp import FastMCP
import os

# Initialize the MCP server with the name "FileSystem"
mcp = FastMCP("FileSystem")

# --- Prompts (Optional, but good to have a system prompt for consistency) ---
@mcp.prompt()
def system_prompt() -> str:
    """
    System prompt description.
    This prompt provides general instructions for an AI assistant focused on file operations.
    """
    return """
    You are an AI assistant specialized in managing local files. Use the provided tools if needed.
    """

# --- Tools ---
@mcp.tool()
def create_text_file(filename: str, content: str) -> str:
    """
    Creates a text file with the given filename and writes the content to it.
    The file will be saved in the directory where this MCP server is running.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        absolute_path = os.path.abspath(filename)
        print(f"MCP Server: Created file '{filename}' at '{absolute_path}'")
        return f"Successfully created file '{filename}' at '{absolute_path}' with content: '{content}'"
    except Exception as e:
        print(f"MCP Server: Error creating file '{filename}': {e}")
        return f"Error creating file '{filename}': {e}"

@mcp.tool()
def append_to_text_file(filename: str, content_to_append: str) -> str:
    """
    Appends content to an existing text file.
    """
    try:
        with open(filename, 'a') as f:
            f.write(content_to_append)
        absolute_path = os.path.abspath(filename)
        print(f"MCP Server: Appended to file '{filename}' at '{absolute_path}'")
        return f"Successfully appended to file '{filename}' at '{absolute_path}'. Appended content: '{content_to_append}'"
    except FileNotFoundError:
        print(f"MCP Server: Error: File '{filename}' not found for appending.")
        return f"Error: File '{filename}' not found. Use 'create_text_file' first."
    except Exception as e:
        print(f"MCP Server: Error appending to file '{filename}': {e}")
        return f"Error appending to file '{filename}': {e}"

@mcp.tool()
def read_text_file(filename: str) -> str:
    """
    Reads the content of a text file and returns it.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        print(f"MCP Server: Read file '{filename}'")
        return f"Content of '{filename}':\n{content}"
    except FileNotFoundError:
        print(f"MCP Server: Error: File '{filename}' not found for reading.")
        return f"Error: File '{filename}' not found."
    except Exception as e:
        print(f"MCP Server: Error reading file '{filename}': {e}")
        return f"Error reading file '{filename}': {e}"

# Main execution block: runs the server.
if __name__ == "__main__":
    print("File System MCP Server: Starting...")
    # This server will run via standard I/O, allowing the LangGraph agent to communicate with it as a subprocess.
    mcp.run()
    print("File System MCP Server: Shutting down.")