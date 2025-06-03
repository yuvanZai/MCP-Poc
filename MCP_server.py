# math_server.py
from mcp.server.fastmcp import FastMCP
from groq import Groq
import os

mcp = FastMCP("TextTools")
GROQ_API_KEY = "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" # Replace with your key

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq(api_key=GROQ_API_KEY)

@mcp.tool()
def Generate_story(text: str) -> str:
    """Generate story from the given text using Groq LLM."""
    print(f"[MCP] Tool called: Generate_story | input: {text}")
    prompt = f"generate and make story from the following text:\n\n{text}"
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

@mcp.tool()
def code_review(code: str) -> str:
    """Provide a code review for the given code using Groq LLM."""
    print(f"[MCP] Tool called: code_review | input: {code}")
    prompt = f"Please review the following code and provide suggestions for improvement:\n\n{code}"
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print("Starting MCP server with Groq LLM...")
    mcp.run(transport="stdio")