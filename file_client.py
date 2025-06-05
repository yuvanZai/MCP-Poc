# langgraph_mcp_agent.py
# This file integrates a FileSystem MCP server with LangGraph
# to create an AI agent capable of using file system tools.

from typing import List, Annotated, TypedDict
import asyncio
import requests
import json

# LangChain and LangGraph imports
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatResult, Generation, ChatGeneration
from langchain_core.runnables import Runnable

# MCP client and adapter imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt

# --- Custom LLM Integration (Your Vegas LLM) ---
# Assuming you have a config.py with VEGAS_GENAI_TOKEN and vegas_end_point
try:
    from config import VEGAS_GENAI_TOKEN, vegas_end_point
except ImportError:
    print("WARNING: config.py not found. Please create it with VEGAS_GENAI_TOKEN and vegas_end_point.")
    VEGAS_GENAI_TOKEN = "YOUR_VEGAS_API_KEY" # Replace with your actual key if config.py isn't used
    vegas_end_point = "YOUR_VEGAS_ENDPOINT_URL" # Replace with your actual URL if config.py isn't used


class VegasLLM(BaseChatModel):
    """Custom LangChain wrapper for your Vegas LLM."""

    def _generate(self, messages: List[AnyMessage], stop=None) -> ChatResult:
        print("VegasLLM: Generating response...")
        system_instructions_parts = []

        system_instructions_parts.append(
            "You are an AI assistant specialized in managing local files. If you need to perform a file operation, use the available tools.")

        for msg in messages:
            if isinstance(msg, HumanMessage):
                system_instructions_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_call_strs = []
                    for tool_call in msg.tool_calls:
                        # Ensure 'id' is always present for ToolMessage linking later
                        tool_call_id = tool_call.get("id", f"call_{tool_call['name']}_{asyncio.get_event_loop().time()}")
                        tool_call_strs.append(
                            f'{{"name": "{tool_call["name"]}", "args": {json.dumps(tool_call["args"])}, "id": "{tool_call_id}"}}'
                        )
                    system_instructions_parts.append(f"AI suggested tool calls: [{', '.join(tool_call_strs)}]")
                if msg.content:
                    system_instructions_parts.append(f"AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                # Ensure we handle the 'tool_call_id' for correct association
                tool_id_info = f" (tool_call_id: {msg.tool_call_id})" if msg.tool_call_id else ""
                system_instructions_parts.append(f"Tool output{tool_id_info}: {msg.content}")

        system_instructions = "\n".join(system_instructions_parts)
        print(f"VegasLLM: Sending to Vegas LLM with system_instructions:\n{system_instructions[:500]}...")

        response_bytes = self._call_vegas_2(system_instructions)
        response_str = response_bytes.decode('utf-8')
        response_json = json.loads(response_str)
        prediction_text = response_json.get('prediction', 'No prediction found.')
        print(f"VegasLLM: Raw prediction from Vegas LLM: {prediction_text[:500]}...")

        ai_message: BaseMessage

        try:
            parsed_prediction = json.loads(prediction_text)
            if isinstance(parsed_prediction, dict) and "tool_calls" in parsed_prediction:
                langchain_tool_calls = []
                for tc_data in parsed_prediction["tool_calls"]:
                    # Generate a unique ID if not provided by the LLM, crucial for LangGraph's ToolNode
                    tool_id = tc_data.get("id", f"call_{tc_data.get('name', 'unknown')}_{asyncio.get_event_loop().time()}")
                    langchain_tool_calls.append({
                        "name": tc_data.get("name"),
                        "args": tc_data.get("args", {}),
                        "id": tool_id
                    })
                ai_message = AIMessage(content="", tool_calls=langchain_tool_calls)
                print(f"VegasLLM: Parsed as tool call: {ai_message.tool_calls}")
            else:
                ai_message = AIMessage(content=prediction_text)
                print(f"VegasLLM: Parsed as text response: {ai_message.content}")
        except json.JSONDecodeError:
            ai_message = AIMessage(content=prediction_text)
            print(f"VegasLLM: Parsed as plain text (not JSON): {ai_message.content}")

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _call_vegas_2(self, system_instructions):
        payload = {
            "useCase": "code_conv",
            "contextId": "test",
            "preSeed_injection_map": {
                "system_instructions": system_instructions,
            },
        }
        headers = {'X-apikey': VEGAS_GENAI_TOKEN, 'Content-Type': 'application/json'}
        response = requests.post(vegas_end_point, json=payload, headers=headers)
        llm_response = response.content
        return llm_response

    @property
    def _llm_type(self) -> str:
        return "vegas_llm"

    def get_num_tokens(self) -> int:
        return 0

    def bind_tools(
            self,
            tools: List[dict],
            **kwargs,
    ) -> Runnable:
        self.bound_tools = tools
        return self


# --- MultiServerMCPClient Setup ---
# Now configured to only manage the "filesystem" server
client = MultiServerMCPClient(
    {
        "filesystem": {
            "command": "python",
            "args": ["file_system_mcp_server.py"], # Command to run the filesystem server
            "transport": "stdio",
        }
    }
)

# --- State Management for LangGraph ---
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# --- Graph Creation Function ---
# This function now only takes the filesystem_session
async def create_graph(filesystem_session):
    llm = VegasLLM()

    # Load tools ONLY from the filesystem MCP server
    filesystem_tools = await load_mcp_tools(filesystem_session)
    tools = filesystem_tools

    tool_definitions_for_llm = []
    for t in tools:
        tool_info = {"name": t.name, "description": t.description}
        tool_info["parameters"] = t.args if hasattr(t, 'args') else {}
        tool_definitions_for_llm.append(tool_info)

    # Load the system prompt from the "filesystem" MCP server
    # Assuming file_system_mcp_server.py also has a 'system_prompt'
    system_prompt_messages = await load_mcp_prompt(filesystem_session, "system_prompt")
    system_prompt_content = system_prompt_messages[0].content

    tools_json_string = json.dumps(tool_definitions_for_llm, indent=2)
    escaped_tools_json_string = tools_json_string.replace("{", "{{").replace("}", "}}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content + "\n\n" +
         "Available tools:\n" +
         escaped_tools_json_string +
         "\n\n" +
         "If you need to use a tool, respond with a JSON object like this example: " +
         '{{"tool_calls": [{{"name": "tool_name", "args": {{"arg1": "value1"}}, "id": "call_YOUR_ID"}}]}}' +
         "\n" +
         "YOUR_ID can be any unique string for this specific tool call." +
         "\n\n" +
         "**If asked about your capabilities or the tools you have, you should describe the tools available.**" +
         "\n\n" +
         "**Important:** If the user asks to create a file, save text to a file, or write to a new file, use the `create_text_file` tool. " +
         "If the user asks to add content to an existing file or append text, use the `append_to_text_file` tool. " +
         "If the user asks to read a file or show its content, use the `read_text_file` tool." # New instruction for read_text_file
        ),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm

    def chat_node(state: State) -> State:
        result = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [result]} # Ensure it's a list containing the message

    graph_builder = StateGraph(State)

    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))

    graph_builder.add_edge(START, "chat_node")

    graph_builder.add_conditional_edges(
        "chat_node",
        tools_condition,
        {"tools": "tool_node", "__end__": END}
    )

    graph_builder.add_edge("tool_node", "chat_node")

    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph

# --- Main Execution Function ---
async def main():
    config = {"configurable": {"thread_id": 1234}}

    print("Main: Creating MCP client session for 'filesystem' server...")
    async with client.session("filesystem") as filesystem_session: # Only filesystem session now
        print("Main: MCP 'filesystem' session created. Creating LangGraph agent...")
        agent = await create_graph(filesystem_session) # Pass only filesystem_session
        print("LangGraph agent initialized. Type 'exit' to quit.")

        while True:
            try:
                message = input("User: ")
                if message.lower() == 'exit':
                    print("Exiting...")
                    break

                print(f"Main: Invoking agent with message: '{message}'")
                response = await agent.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)

                print("\n--- Agent Response Trace ---")
                for msg in response["messages"]:
                    if isinstance(msg, HumanMessage):
                        print(f"Human: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            print(f"AI (Tool Call): {msg.tool_calls}")
                        if msg.content:
                            print(f"AI (Content): {msg.content}")
                    elif isinstance(msg, ToolMessage):
                        print(f"Tool Output (from call_id: {msg.tool_call_id or 'N/A'}): {msg.content}")
                print("--- End Trace ---\n")

                final_ai_message = next((msg for msg in reversed(response["messages"]) if isinstance(msg, AIMessage) and msg.content), None)
                if final_ai_message:
                    print("AI: " + final_ai_message.content)
                else:
                    print("AI: (No final content message from AI)")

            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())