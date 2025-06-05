# langgraph_mcp_agent.py
from typing import List, Annotated, TypedDict
import asyncio
import requests
import json

# LangChain and LangGraph imports
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver  # Corrected import for MemorySaver
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
    VEGAS_GENAI_TOKEN = "YOUR_VEGAS_API_KEY"
    vegas_end_point = "YOUR_VEGAS_ENDPOINT_URL"


class VegasLLM(BaseChatModel):
    """Custom LangChain wrapper for your Vegas LLM."""

    def _generate(self, messages: List[AnyMessage], stop=None) -> ChatResult:
        print("VegasLLM: Generating response...")
        system_instructions_parts = []

        # This part constructs the prompt for your custom LLM, including tool definitions.
        # It's crucial that your Vegas LLM understands this format to call tools.
        # Adjust this based on how your Vegas LLM is trained to receive tool definitions and calls.

        # Add a general instruction for the AI to use tools
        system_instructions_parts.append(
            "You are an AI assistant. If you need to perform a calculation, use the available tools.")

        # Iterate through messages to build conversation history for the LLM
        for msg in messages:
            if isinstance(msg, HumanMessage):
                system_instructions_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # Represent tool calls in a way your LLM understands
                    tool_call_strs = []
                    for tool_call in msg.tool_calls:
                        tool_call_strs.append(
                            f'{{"name": "{tool_call.name}", "args": {json.dumps(tool_call.args)}}}, "id": "{tool_call.id}"'
                        )
                    system_instructions_parts.append(f"AI suggested tool calls: [{', '.join(tool_call_strs)}]")
                if msg.content:
                    system_instructions_parts.append(f"AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                system_instructions_parts.append(
                    f"Tool output for tool_call_id {msg.tool_call_id or 'unknown'}: {msg.content}")

        system_instructions = "\n".join(system_instructions_parts)
        print(f"VegasLLM: Sending to Vegas LLM with system_instructions:\n{system_instructions[:500]}...")

        response_bytes = self._call_vegas_2(system_instructions)
        response_str = response_bytes.decode('utf-8')
        response_json = json.loads(response_str)
        prediction_text = response_json.get('prediction', 'No prediction found.')
        print(f"VegasLLM: Raw prediction from Vegas LLM: {prediction_text[:500]}...")

        ai_message: BaseMessage

        # Attempt to parse the prediction as a tool call or a regular text response
        try:
            parsed_prediction = json.loads(prediction_text)
            if isinstance(parsed_prediction, dict) and "tool_calls" in parsed_prediction:
                # Ensure the tool_calls structure matches LangChain's expected format
                # This might need adjustment based on how your Vegas LLM returns tool calls
                langchain_tool_calls = []
                for tc_data in parsed_prediction["tool_calls"]:
                    langchain_tool_calls.append({
                        "name": tc_data.get("name"),
                        "args": tc_data.get("args", {}),
                        "id": tc_data.get("id", f"call_{tc_data.get('name')}_{asyncio.get_event_loop().time()}")
                        # Generate ID if missing
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
        headers = {'X-apikey': VEGAS_GENAI_TOKEN,
                   'Content-Type': 'application/json'}  # Corrected 'context-Type' to 'Content-Type'
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
        # In a custom LLM, you might pass tool definitions to the LLM here
        # For simplicity, we're assuming the system prompt handles this.
        # This method is crucial if your LLM expects explicit tool schemas via API.
        self.bound_tools = tools  # Store tools for later use if needed
        return self


# --- MultiServerMCPClient Setup ---
# Only include the 'math' server, as the 'bmi' server setup is for HTTP and
# would require a separate process and URL. For this example, we'll focus
# on the stdio-based math server which the client can manage as a subprocess.
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["math_mcp_server.py"],  # Command to run the math server
            "transport": "stdio",
        },
        # If you were to add BMI, it would need a separate process running first:
        # "bmi": {
        #     "url": "http://localhost:8000/mcp",
        #     "transport": "streamable_http",
        # }
    }
)


# --- State Management for LangGraph ---
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# --- Graph Creation Function ---
async def create_graph(math_session):
    llm = VegasLLM()

    # Load tools from the math MCP server
    math_tools = await load_mcp_tools(math_session)
    tools = math_tools  # If you had more servers, you'd combine like: tools = math_tools + bmi_tools

    # Prepare tool definitions for the LLM to understand
    tool_definitions_for_llm = []
    for t in tools:
        tool_info = {"name": t.name, "description": t.description}
        # MCP tools have an 'args' attribute for their signature
        tool_info["parameters"] = t.args if hasattr(t, 'args') else {}
        tool_definitions_for_llm.append(tool_info)

    # Load the system prompt from the "math" MCP server
    system_prompt_messages = await load_mcp_prompt(math_session, "system_prompt")
    system_prompt_content = system_prompt_messages[0].content

    # Embed tool definitions into the system prompt for the Vegas LLM
    # This is a common pattern for custom LLMs that don't have native tool calling.
    tools_json_string = json.dumps(tool_definitions_for_llm, indent=2)
    # Escape curly braces for f-string or use .format()
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
         "**If asked about your capabilities or the tools you have, you should describe the tools available.**"),
        MessagesPlaceholder("messages")
    ])

    # Chain the prompt template with the custom LLM
    chat_llm = prompt_template | llm

    # --- Nodes for the Graph ---
    def chat_node(state: State) -> State:
        """
        The chat node invokes the LLM to generate a response.
        It updates the 'messages' in the state with the LLM's output.
        """
        # Invoke the LLM with the current conversation messages
        response_message = chat_llm.invoke({"messages": state["messages"]})
        # The invoke method returns a BaseMessage, so we wrap it in a list for the state
        return {"messages": [response_message]}

    # --- Building the LangGraph ---
    graph_builder = StateGraph(State)

    # Add nodes to the graph
    graph_builder.add_node("chat_node", chat_node)
    # ToolNode automatically handles tool execution based on LLM output
    graph_builder.add_node("tool_node", ToolNode(tools=tools))

    # Define edges (transitions) between nodes
    # Start the graph by going to the chat_node
    graph_builder.add_edge(START, "chat_node")

    # Conditional edge from chat_node:
    # If the LLM output suggests using a tool, go to "tool_node".
    # Otherwise, if the LLM output is a final answer, end the graph ("__end__").
    graph_builder.add_conditional_edges(
        "chat_node",
        tools_condition,  # LangGraph's prebuilt condition to check for tool calls
        {"tools": "tool_node", "__end__": END}
    )

    # After a tool is executed, return to the chat_node to allow the LLM to process the tool's result
    graph_builder.add_edge("tool_node", "chat_node")

    # Compile the graph with a memory checkpointer for state persistence across turns
    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph

# --- Main Execution Function ---
async def main():
    """
    Main function to run the LangGraph agent and interact with the user.
    """
    # Configuration for the agent (e.g., thread_id for checkpointing)
    config = {"configurable": {"thread_id": 1234}}

    # Use persistent session for the "math" MCP server.
    # This ensures the server remains active throughout the agent's lifetime.
    print("Main: Creating MCP client session for 'math' server...")
    async with client.session("math") as math_session:
        print("Main: MCP 'math' session created. Creating LangGraph agent...")
        # Create the LangGraph agent with the established MCP session
        agent = await create_graph(math_session)
        print("LangGraph agent initialized. Type 'exit' to quit.")

        # Start the interaction loop
        while True:
            try:
                message = input("User: ")
                if message.lower() == 'exit':
                    print("Exiting...")
                    break

                # Invoke the agent with the user's message
                # The input to a LangGraph agent is typically a dictionary matching its State type.
                # Here, we're adding the user's message as a HumanMessage.
                print(f"Main: Invoking agent with message: '{message}'")
                response = await agent.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)

                # Print the full conversation trace for debugging and understanding flow
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

                # Print the final AI response (the last message in the trace)
                final_ai_message = next(
                    (msg for msg in reversed(response["messages"]) if isinstance(msg, AIMessage) and msg.content), None)
                if final_ai_message:
                    print("AI: " + final_ai_message.content)
                else:
                    print("AI: (No final content message from AI)")

            except Exception as e:
                print(f"An error occurred: {e}")
                # You might want to add more robust error handling or logging here.


# Entry point for running the script
if __name__ == "__main__":
    asyncio.run(main())