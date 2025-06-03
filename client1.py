import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

GROQ_API_KEY = "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" # Replace with your key

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

class MCPLangGraphAgent:
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # More stable model
            temperature=0.1
        )
        self.client = None
        self.graph = None
        
    async def setup(self):
        """Initialize MCP client and build graph"""
        # Setup MCP client
        self.client = MultiServerMCPClient({
            "demo_server": {
                "command": "python",
                "args": [self.mcp_server_path],
                "transport": "stdio",
            }
        })
        
        # Get tools from MCP server
        tools = await self.client.get_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Build the graph
        await self._build_graph(tools)
        
    async def _build_graph(self, tools):
        """Build the LangGraph workflow"""
        def call_model(state: MessagesState):
            """Node that calls the LLM with available tools"""
            messages = state["messages"]
            
            # Bind tools to the model
            model_with_tools = self.llm.bind_tools(tools)
            response = model_with_tools.invoke(messages)
            
            return {"messages": [response]}
        
        # Create the graph
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("agent", call_model)
        builder.add_node("tools", ToolNode(tools))
        
        # Set up edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            tools_condition,
        )
        builder.add_edge("tools", "agent")
        
        # Compile the graph
        self.graph = builder.compile()
        
    async def chat(self, message: str):
        """Send a message to the agent"""
        if not self.graph:
            await self.setup()
            
        try:
            response = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config={"recursion_limit": 10}
            )
            
            return response["messages"][-1].content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()

# Example usage
async def demo():
    # Update this path to your actual MCP server file
    server_path = "D:\\con ai\\MCP_serv2.py"
    
    agent = MCPLangGraphAgent(server_path)
    
    try:
        print("Setting up agent...")
        await agent.setup()
        print("Agent ready!")
        
        # Test different capabilities
        test_messages = [
            "Can you calculate 15 * 7 + 23?",
            "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?",
            "Can you reverse the text 'Hello World'?",
            "What's 100 divided by 4, and then count the words in 'artificial intelligence is amazing'?",
        ]
        
        for msg in test_messages:
            print(f"\n🤖 Human: {msg}")
            response = await agent.chat(msg)
            print(f"🤖 Agent: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(demo())