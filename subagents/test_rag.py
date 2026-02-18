from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    file_path = r"E:\Langchain-projects\StartAI\subagents\Hazem_Gamal_CV_.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    embedding_function = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_store = Chroma(
        "my_first_collection",
        embedding_function=embedding_function,
        persist_directory="./chroma_langchain_db",
    )
    vector_store.add_documents(documents=all_splits)
    print(vector_store.similarity_search("what is my GPA ?", k=1))
