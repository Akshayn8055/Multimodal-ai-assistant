import faiss
import numpy as np
import json
import os
import shutil

MEMORY_DB_FILE = "data/memory.json"
VECTOR_DB_FILE = "data/memory_index.faiss"
VECTOR_DIMENSION = 512  # Size of vector embeddings

class AIMemory:
    def __init__(self):
        """Initialize AI memory system with FAISS and JSON storage."""
        self.memory = self.load_memory()
        self.index = self.load_vector_db()

    def load_memory(self):
        """Load memory from a JSON file, with error handling."""
        os.makedirs("data", exist_ok=True)  # Create directory if missing
        
        if not os.path.exists(MEMORY_DB_FILE):
            print("Creating new memory.json file...")
            with open(MEMORY_DB_FILE, "w") as file:
                json.dump([], file, indent=4)

        try:
            with open(MEMORY_DB_FILE, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}. Resetting memory.")
            # Reset the memory file if corrupted
            with open(MEMORY_DB_FILE, 'w') as file:
                json.dump([], file)
            return []
        except Exception as e:
            print(f"Error loading memory: {e}")
            return []

    def save_memory(self):
        """Save memory entries to JSON file with atomic write and backup."""
        try:
            # Backup existing memory file
            if os.path.exists(MEMORY_DB_FILE):
                backup_file = MEMORY_DB_FILE + ".bak"
                shutil.copy(MEMORY_DB_FILE, backup_file)

            # Write to a temporary file first
            temp_file = MEMORY_DB_FILE + ".tmp"
            with open(temp_file, 'w') as file:
                json.dump(self.memory, file, indent=4)

            # Atomically move the temp file to the target file
            os.replace(temp_file, MEMORY_DB_FILE)
        except Exception as e:
            print(f"Error saving memory: {e}")
            # Restore from backup in case of error
            if os.path.exists(MEMORY_DB_FILE + ".bak"):
                shutil.copy(MEMORY_DB_FILE + ".bak", MEMORY_DB_FILE)

    def load_vector_db(self):
        """Load or create FAISS vector database."""
        if os.path.exists(VECTOR_DB_FILE):
            index = faiss.read_index(VECTOR_DB_FILE)
        else:
            index = faiss.IndexFlatL2(VECTOR_DIMENSION)  # L2 distance for similarity search
        return index

    def save_vector_db(self):
        """Save FAISS index to file."""
        faiss.write_index(self.index, VECTOR_DB_FILE)

    def add_memory(self, text, embedding):
        """Store a new memory with its vector embedding."""
        embedding = np.array(embedding, dtype=np.float32)  # Ensure correct dtype
        if embedding.shape != (VECTOR_DIMENSION,):
            raise ValueError(f"Embedding must have shape ({VECTOR_DIMENSION},), but got {embedding.shape}")

        memory_entry = {"text": text, "vector": embedding.tolist()}
        self.memory.append(memory_entry)
        self.index.add(embedding.reshape(1, -1))  # Ensure correct shape for FAISS
        self.save_memory()
        self.save_vector_db()

    def search_memory(self, query_embedding, top_k=3):
        """Retrieve the most relevant past memories based on similarity search."""
        if self.index.ntotal == 0:
            print("Memory is empty. No results found.")
            return []

        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        retrieved_memories = []
        for i in range(top_k):
            if indices[0][i] < len(self.memory):  # Ensure valid index
                retrieved_memories.append(self.memory[indices[0][i]]["text"])

        return retrieved_memories

# Example Usage
if __name__ == "__main__":
    ai_memory = AIMemory()
    
    # Example: Adding a new memory
    test_embedding = np.random.rand(VECTOR_DIMENSION).astype(np.float32)  # Random embedding
    ai_memory.add_memory("The user likes AI ethics.", test_embedding)

    # Example: Searching similar memories
    search_embedding = np.random.rand(VECTOR_DIMENSION).astype(np.float32)  # New random vector
    results = ai_memory.search_memory(search_embedding)
    print("Retrieved Memories:", results)
