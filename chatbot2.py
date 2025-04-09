import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from rich.console import Console
from rich.prompt import Prompt
import os
import pandas as pd
from PyPDF2 import PdfReader
import json
import numpy as np

# PostgreSQL Connection Setup
DB_CONFIG = {
    "dbname": "chatbot",
    "user": "postgres",
    "password": "<password>",
    "host": "localhost",
    "port": 5432,
}

OLLAMA_API_URL = "http://localhost:11434/api"

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
            embedding vector(1536) NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# Generate embeddings using Ollama API
def generate_embedding(text):
    response = requests.post(
        f"{OLLAMA_API_URL}/embeddings",
        json={"model": "llama3", "prompt": text}
    )
    if response.status_code == 200:
        embedding = response.json().get("embedding", [])
        if len(embedding) == 1536:
            return embedding
        console.print("[bold red]Error:[/bold red] Embedding size mismatch.")
    return []


# Store embeddings in PostgreSQL
def store_embedding(text):
    embedding = generate_embedding(text)
    if embedding:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO embeddings (text, embedding) VALUES (%s, %s)", (text, np.array(embedding).tolist()))
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


# Get response from Ollama API
def get_ollama_response(prompt):
    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={"model": "llama3", "prompt": prompt},
        stream=True  # Enable streaming
    )

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


# Process file and store embeddings
def process_file(file_path):
    extracted_text = ""

    if file_path.startswith("https://drive.google.com"):  # Google Drive file
        console.print("[bold red]Error:[/bold red] Google Drive file reading not supported without downloading.")
        return
    elif os.path.exists(file_path):  # Local file
        console.print(f"[bold yellow]Processing file:[/bold yellow] {file_path}")
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                extracted_text = file.read()
        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            extracted_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            extracted_text = df.to_string()

    if not extracted_text:
        console.print("[bold red]Error:[/bold red] No text found in file.")
        return

    # Split text into chunks and embed them
    chunk_size = 500  # Adjust chunk size as needed
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

        # Retrieve similar text if available
        similar_text = retrieve_similar_text(user_input)

        # Generate response
        context = f"Similar context: {similar_text}" if similar_text else "No prior knowledge."
        final_prompt = f"{context}\nUser: {user_input}"
        response = get_ollama_response(final_prompt)

        console.print(f"[bold magenta]Bot:[/bold magenta] {response}")

        # Store user input for future reference
        store_embedding(user_input)


# Run Setup and Start Chat
if __name__ == "__main__":
    setup_database()
    chat()
