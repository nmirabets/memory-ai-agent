import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime, timezone


load_dotenv()

# Initialize Pinecone for vector database
pc = Pinecone(os.getenv("PINECONE_API_KEY"))
# Initialize the vector database index
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
# Initialize OpenAI for embeddings 
client = OpenAI()

# Define the tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                    },
                    "required": ["location"]
                },
            },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save a memory to the vector database",
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["memories"]
            },
        },
    },
]


def get_embeddings(string_to_embed):
    response = client.embeddings.create(
        input=string_to_embed,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def save_memory(memories):
    for memory in memories:
        # Step 1: Embed the memory
        vector = get_embeddings(memory)
        # Step 2: Build the vector document to be stored
        user_id = "1234"
        current_time = datetime.now(tz=timezone.utc)

        documents = [
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "user_id": user_id,
                    "timestamp": str(current_time),
                    "payload": memory,
                },
            }
        ]
        # Step 3: Store the vector document in the vector database
        index.upsert(
            vectors=documents,
            namespace=os.getenv("PINECONE_NAMESPACE")
        )
    return f"{len(memories)} memories saved successfully"

def load_memories(prompt):
    user_id = "1234"
    top_k = 2
    vector = get_embeddings(prompt)
    response = index.query(
        vector=vector,
        filter={
            "user_id": {"$eq": user_id},
        },
        namespace=os.getenv("PINECONE_NAMESPACE"),
        include_metadata=True,
        top_k=top_k,
    )
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"]["payload"] for m in matches]

    return memories