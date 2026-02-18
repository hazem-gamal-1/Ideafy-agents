import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Use PersistentClient for on-disk storage
client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# get_or_create_collection avoids errors on re-runs
collection = client.get_or_create_collection(
    name="docs", embedding_function=embedding_function
)

collection.add(
    ids=["1", "2", "3","4"],
    documents=[
        "Chroma is an open-source vector database",
        "Vector databases store embeddings",
        "RAG uses vector search with LLMs",
        "my name is hazem",
    ],
)

results = collection.query(query_texts=["What is your name ?"], n_results=1)

print(results["documents"])
