import os
import uuid
import requests
import sys

# Set up the base URL for LlamaStack API
BASE_URL = f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"


# Function to register a vector database
def register_vector_db():
    vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
    provider_id = "faiss-provider"  # Replace with your provider ID
    embedding_model = "all-MiniLM-L6-v2"
    embedding_dimension = 384

    # Prepare the payload for the vector DB registration
    payload = {
        "vector_db_id": vector_db_id,
        "provider_id": provider_id,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
    }

    # Register the vector DB
    response = requests.post(f"{BASE_URL}/vector_dbs/register", json=payload)

    if response.status_code == 200:
        print(f"Vector database registered successfully: {vector_db_id}")
        return vector_db_id
    else:
        print(f"Failed to register vector database. Status code: {response.status_code}")
        print(response.text)
        sys.exit(1)


# Function to insert documents into the vector database
# we can put documents related to pizzas to make it work like pizza bot.
def insert_documents(vector_db_id):
    documents = [
        {
            "document_id": f"num-{i}",
            "content": f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            "mime_type": "text/plain",
            "metadata": {},
        }
        for i, url in enumerate([
            "chat.rst", "llama3.rst", "memory_optimizations.rst", "lora_finetune.rst"
        ])
    ]

    # Prepare the payload for inserting documents
    payload = {
        "documents": documents,
        "vector_db_id": vector_db_id,
        "chunk_size_in_tokens": 512
    }

    # Insert documents into the vector database
    response = requests.post(f"{BASE_URL}/tools/rag_tool/insert", json=payload)

    if response.status_code == 200:
        print("Documents inserted successfully.")
    else:
        print(f"Failed to insert documents. Status code: {response.status_code}")
        print(response.text)


# Function to create an agent
def create_agent(vector_db_id):
    # Prepare the agent creation payload
    agent_payload = {
        "model": os.environ["INFERENCE_MODEL"],  # Replace with your model name
        "instructions": "You are a helpful assistant",
        "enable_session_persistence": False,
        "tools": [{
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        }],
    }

    # Create the agent
    response = requests.post(f"{BASE_URL}/agents", json=agent_payload)

    if response.status_code == 200:
        print("Agent created successfully.")
        return response.json()["agent_id"]
    else:
        print(f"Failed to create agent. Status code: {response.status_code}")
        print(response.text)
        sys.exit(1)


# Function to create a session for the agent
def create_session(agent_id):
    # Prepare the session creation payload
    session_payload = {"session_name": "test-session"}

    # Create the session
    response = requests.post(f"{BASE_URL}/agents/{agent_id}/sessions", json=session_payload)

    if response.status_code == 200:
        print("Session created successfully.")
        return response.json()["session_id"]
    else:
        print(f"Failed to create session. Status code: {response.status_code}")
        print(response.text)
        sys.exit(1)


# Function to send a user prompt to the agent and get a response
def chat_with_agent(agent_id, session_id):
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the chatbot...")
            break

        # Prepare the user prompt payload
        turn_payload = {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }

        # Send the user prompt to the agent
        response = requests.post(f"{BASE_URL}/agents/{agent_id}/sessions/{session_id}/turns", json=turn_payload)

        if response.status_code == 200:
            agent_response = response.json()
            print(f"Agent: {agent_response['response']}")
        else:
            print(f"Failed to get response. Status code: {response.status_code}")
            print(response.text)


def main():
    # Step 1: Register vector database
    vector_db_id = register_vector_db()

    # Step 2: Insert documents into the vector database
    insert_documents(vector_db_id)

    # Step 3: Create an agent
    agent_id = create_agent(vector_db_id)

    # Step 4: Create a session for the agent
    session_id = create_session(agent_id)

    # Step 5: Start chatting with the agent
    chat_with_agent(agent_id, session_id)


if __name__ == "__main__":
    main()
