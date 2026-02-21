from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ContextRetrieval:
    def __init__(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        all_splits = text_splitter.split_documents(docs)
        embedding_function = OllamaEmbeddings(model="nomic-embed-text:latest")
        self._vector_store = Chroma(
            collection_name="my_first_collection",
            embedding_function=embedding_function,
            persist_directory="./chroma_langchain_db",
        )
        self._vector_store.add_documents(documents=all_splits)

    def retrieve_context(self, query):
        """Search internal startup knowledge base for relevant context."""
        docs = self.vector_store.similarity_search(query, k=4)
        return "\n\n".join(d.page_content for d in docs)
