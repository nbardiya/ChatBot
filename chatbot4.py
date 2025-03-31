import requests
import json
import os
import pandas as pd
from PyPDF2 import PdfReader
from rich.console import Console
from rich.prompt import Prompt
from pdfminer.high_level import extract_text

# Constants
OLLAMA_API_URL = "http://localhost:11434/api"
CUSTOM_MODEL_NAME = None  # Global model name
console = Console()
MODEL_REGISTRY_FILE = "model_registry.json"  # Store created model names


# Save the created model globally
def save_model_name(model_name):
    global CUSTOM_MODEL_NAME
    CUSTOM_MODEL_NAME = model_name
    with open(MODEL_REGISTRY_FILE, "w") as f:
        json.dump({"model_name": model_name}, f)
    console.print(f"[bold green]Model '{model_name}' registered globally![/bold green]")


# Load the last created model
def load_model_name():
    global CUSTOM_MODEL_NAME
    if os.path.exists(MODEL_REGISTRY_FILE):
        with open(MODEL_REGISTRY_FILE, "r") as f:
            data = json.load(f)
            CUSTOM_MODEL_NAME = data.get("model_name")
            console.print(f"[bold yellow]Using registered model:[/bold yellow] {CUSTOM_MODEL_NAME}")


# Extract text from a file
def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        return None

    console.print(f"[bold yellow]Reading file:[/bold yellow] {file_path}")

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif file_path.endswith(".pdf"):
        try:
            extracted_text = extract_text(file_path)
            return extracted_text.strip() if extracted_text else None
        except Exception as e:
            print(f"Error extracting text using PDFMiner: {e}")
            return None
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_string()

    return None


# Create model using Ollama API
def create_model(file_path):
    system_content = extract_text_from_file(file_path)

    if not system_content:
        console.print("[bold red]Error:[/bold red] No valid content to train model.")
        return

    model_name = os.path.splitext(os.path.basename(file_path))[0] + "_model"

    payload = {
        "name": model_name,
        "system": system_content,
        "from": "llama3.2"
    }

    response = requests.post(f"{OLLAMA_API_URL}/create", json=payload)

    if response.status_code == 200:
        console.print(f"[bold green]Model '{model_name}' created successfully![/bold green] {response.text}")
        save_model_name(model_name)  # Register the model globally
    else:
        console.print(f"[bold red]Error creating model:[/bold red] {response.text}")


def load_model():
    if CUSTOM_MODEL_NAME is None:
        console.print("[bold red]Error:[/bold red] No model found. Create a model first.")
        return

    response = requests.post(f"{OLLAMA_API_URL}/pull", json={"name": CUSTOM_MODEL_NAME})
    if response.status_code == 200:
        console.print(f"[bold green]Model '{CUSTOM_MODEL_NAME}' loaded successfully![/bold green]")
    else:
        console.print(f"[bold red]Error loading model:[/bold red] {response.text}")


# Get response from model
def get_custom_model_response(prompt):
    if CUSTOM_MODEL_NAME is None:
        console.print("[bold red]Error:[/bold red] No model is set. Create a model first.")
        return "No model available."

    payload = {
        "model": CUSTOM_MODEL_NAME,
        "prompt": prompt
    }

    response = requests.post(f"{OLLAMA_API_URL}/generate", json=payload)

    full_response = ""

    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode("utf-8"))  # Parse JSON line
                chunk = json_data.get("response", "")
                full_response += chunk  # Append chunk to full response
            except json.JSONDecodeError:
                continue  # Ignore incomplete JSON errors

    return full_response if full_response else "No response from Ollama."


# Main chat loop
def chat():
    load_model_name()  # Load last created model
    console.print(
        "[bold green]Chatbot CLI - Type 'exit' to quit or 'file <path>' to create a model from a file[/bold green]")
    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

        if user_input.lower() == "exit":
            break
        elif user_input.lower().startswith("file "):
            file_path = user_input.split("file ", 1)[1].strip()
            create_model(file_path)
            load_model()  # Ensure the model is loaded
            continue

        response = get_custom_model_response(user_input)
        console.print(f"[bold magenta]Bot:[/bold magenta] {response}")


# Run chat
if __name__ == "__main__":
    chat()
