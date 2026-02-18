import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings


client = chromadb.Client(
    Settings(persist_directory="./chroma_db")  # this folder will store your database
)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="docs", embedding_function=embedding_function
)


collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Chroma is an open-source vector database",
        "Vector databases store embeddings",
        "RAG uses vector search with LLMs",
    ],
    metadatas=[{"topic": "chroma"}, {"topic": "vector-db"}, {"topic": "rag"}],
)
results = collection.query(query_texts=["What is a vector database?"], n_results=1)

print(results["documents"])
