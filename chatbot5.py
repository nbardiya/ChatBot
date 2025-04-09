from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage

# Initialize the client
client = LlamaStackClient(base_url="http://localhost:8321/v1")

# Implement Chat Completion
# To generate responses from the model, use the chat_completion method
def get_chat_response(user_input):
    messages = [UserMessage(content=user_input, role="user")]
    response = client.inference.chat_completion(
        messages=messages,
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        stream=False,
    )
    return response.completion_message.content

# Manage Memory
# To store and retrieve conversational memory:
def add_memory_entry(text):
    client.memory.add(text=text)

def fetch_memory_entries():
    return client.memory.list()

# Perform Safety Checks
# To ensure the safety of user inputs or model outputs:
def perform_safety_check(text):
    response = client.safety.check(text=text)
    return response


# List and Invoke Agents
# To interact with available agents
def list_available_agents():
    return client.inspect.providers()

def invoke_agent(query):
    response = client.tool_runtime.invoke(query=query)
    return response

# List and Use Tools
# To utilize tools within the Llama Stack
def list_available_tools():
    return client.tool_runtime.list_tools()

def use_tool(input_text):
    response = client.tool_runtime.rag_tool.query(input=input_text)
    return response

# Create the CLI Interface
def main():
    print("RAG Chatbot CLI - Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "check memory":
            memories = fetch_memory_entries()
            print("Memory Entries:", memories)
        elif user_input.lower() == "check safety":
            safety_result = perform_safety_check(user_input)
            print("Safety Check Result:", safety_result)
        elif user_input.lower() == "list agents":
            agents = list_available_agents()
            print("Available Agents:", agents)
        elif user_input.lower() == "invoke agent":
            agent_response = invoke_agent(user_input)
            print("Agent Response:", agent_response)
        elif user_input.lower() == "list tools":
            tools = list_available_tools()
            print("Available Tools:", tools)
        elif user_input.lower() == "use tool":
            tool_response = use_tool(user_input)
            print("Tool Response:", tool_response)
        else:
            response = get_chat_response(user_input)
            print("Bot:", response)
            add_memory_entry(user_input)

if __name__ == "__main__":
    main()
