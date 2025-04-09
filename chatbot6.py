import llama_stack_client
import json
from rich.console import Console
from rich.prompt import Prompt

LLAMASTACK_API_URL = "http://localhost:8321/v1"

console = Console()
client = llama_stack_client.LlamaStackClient(base_url=LLAMASTACK_API_URL)

def fetch_memory():
    try:
        response = client.memory.list()
        console.print("Memory:", response)
    except Exception as e:
        console.print("Error fetching memory:", str(e))

def fetch_tools():
    try:
        response = client.tool.list()
        console.print("Tools:", response)
    except Exception as e:
        console.print("Error fetching tools:", str(e))

def fetch_agents():
    try:
        response = client.agent.list()
        console.print("Agents:", response)
    except Exception as e:
        console.print("Error fetching agents:", str(e))

def send_message(user_input):
    try:
        messages = [{"content": user_input, "role": "user"}]
        response = client.inference.generate(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            stream=False
        )
        bot_response = response["choices"][0]["message"]["content"]
        console.print(f"[bold magenta]Bot:[/bold magenta] {bot_response}")
        client.memory.create(text=user_input)
    except Exception as e:
        console.print("Error sending message:", str(e))

def check_safety(user_input):
    try:
        response = client.safety.evaluate(text=user_input)
        console.print("Safety Check:", response)
    except Exception as e:
        console.print("Error checking safety:", str(e))

def use_agent(user_input):
    try:
        response = client.agent.invoke(name="default-agent", input=user_input)
        console.print("Agent Response:", response)
    except Exception as e:
        console.print("Error using agent:", str(e))

def use_tool(user_input):
    try:
        response = client.tool.invoke(name="default-tool", input=user_input)
        console.print("Tool Response:", response)
    except Exception as e:
        console.print("Error using tool:", str(e))

def chat():
    console.print("[bold green]CLI RAG Chatbot - Type 'exit' to quit.[/bold green]")
    fetch_memory()
    fetch_tools()
    fetch_agents()
    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if user_input.lower() == "exit":
            break
        elif user_input.lower().startswith("safety "):
            check_safety(user_input.split("safety ", 1)[1])
        elif user_input.lower().startswith("agent "):
            use_agent(user_input.split("agent ", 1)[1])
        elif user_input.lower().startswith("tool "):
            use_tool(user_input.split("tool ", 1)[1])
        else:
            send_message(user_input)

if __name__ == "__main__":
    chat()
