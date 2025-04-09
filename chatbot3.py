import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from rich.console import Console
from rich.prompt import Prompt
import os
import pandas as pd
from PyPDF2 import PdfReader
import json
import subprocess

# PostgreSQL Connection Setup
DB_CONFIG = {
    "dbname": "chatbot",
    "user": "postgres",
    "password": "<password>",
    "host": "localhost",
    "port": 5432,
}

OLLAMA_API_URL = "http://localhost:11434/api"
CUSTOM_MODEL_NAME = "my_custom_model"
TRAINING_DATA_FILE = "/Users/nbardiya/Downloads/testing_txt.txt"
MODELFILE = "Modelfile"

# Initialize Console UI
console = Console()


# Connect to PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)  # Enable pgvector extension
    return conn


# Create Table for Embeddings
def setup_database():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(4096) NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# Generate embeddings using Ollama API
def generate_embedding(text):
    response = requests.post(
        f"{OLLAMA_API_URL}/embeddings",
        json={"model": CUSTOM_MODEL_NAME, "prompt": text}
    )
    if response.status_code == 200:
        return response.json().get("embedding", [])
    return []


# Store embeddings in PostgreSQL
def store_embedding(text):
    embedding = generate_embedding(text)
    if embedding:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO embeddings (text, embedding) VALUES (%s, %s)", (text, embedding))
        conn.commit()
        conn.close()


# Retrieve most similar text from database
def retrieve_similar_text(query_text):
    query_embedding = generate_embedding(query_text)
    if not query_embedding:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT text FROM embeddings
        ORDER BY embedding <-> %s::vector
        LIMIT 1;
    """, (query_embedding,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None


# Get response from Ollama API using custom model
def get_custom_model_response(prompt):
    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={"model": CUSTOM_MODEL_NAME, "prompt": prompt},
        stream=True
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode("utf-8"))
                chunk = json_data.get("response", "")
                full_response += chunk
            except json.JSONDecodeError:
                continue
    return full_response if full_response else "No response from Ollama."


# Create Modelfile
def create_modelfile():
    with open(MODELFILE, "w") as f:
        f.write(f"""
FROM llama3

SYSTEM "You are a highly reliable, concise, and precise assistant. "
        "When answering user queries, adhere to the following guidelines:\n"
        "1. **Strict Context Usage**: Base your response exclusively on the provided retrieved context.\n"
        "2. **Clarity and Brevity**: Provide clear, focused, and concise responses.\n"
        "3. **Handling Insufficient Information**: If the context lacks sufficient details, state that the necessary information is unavailable.\n"
        "4. **Avoid Inference or Speculation**: Stick to explicit context without speculation.\n"
        "5. **Disambiguation**: If a query is ambiguous, seek clarification before responding.\n"
        "6. **Neutral Tone**: Maintain objectivity in your responses.\n"
        "7. **User-Centric**: Ensure accessibility and clarity for a general audience.\n""

ADAPTER "{TRAINING_DATA_FILE}"
        """)
    console.print("[bold green]Modelfile created successfully.[/bold green]")


# Build and load the custom model
def build_custom_model():
    subprocess.run(["sudo", "ollama", "create", CUSTOM_MODEL_NAME, "-f", MODELFILE], check=True)
    console.print("[bold green]Custom model built and loaded successfully.[/bold green]")


# Process file and store embeddings
def process_file(file_path):
    extracted_text = ""

    if os.path.exists(file_path):
        console.print(f"[bold yellow]Processing file:[/bold yellow] {file_path}")
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                extracted_text = file.read()
        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path, strict=False)
            extracted_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            extracted_text = df.to_string()

    if not extracted_text:
        console.print("[bold red]Error:[/bold red] No text found in file.")
        return

    with open(TRAINING_DATA_FILE, "w") as f:
        f.write(extracted_text)

    create_modelfile()
    build_custom_model()

    chunk_size = 500
    chunks = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]
    for chunk in chunks:
        store_embedding(chunk)
    console.print("[bold green]File processed successfully. Data stored in embeddings.[/bold green]")


# Main Chat Loop
def chat():
    console.print("[bold green]Chatbot CLI - Type 'exit' to quit or 'file <path>' to process a file[/bold green]")
    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

        if user_input.lower() == "exit":
            break
        elif user_input.lower().startswith("file "):
            file_path = user_input.split("file ", 1)[1].strip()
            process_file(file_path)
            continue

        similar_text = retrieve_similar_text(user_input)
        context = f"Similar context: {similar_text}" if similar_text else "No prior knowledge."
        final_prompt = f"{context}\nUser: {user_input}"
        response = get_custom_model_response(final_prompt)

        console.print(f"[bold magenta]Bot:[/bold magenta] {response}")
        store_embedding(user_input)


# Run Setup and Start Chat
if __name__ == "__main__":
    setup_database()
    chat()
