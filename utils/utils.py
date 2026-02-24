from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import tempfile


class ContextRetrieval:
    def __init__(self, file_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        all_splits = text_splitter.split_documents(docs)
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
        )
        self._vector_store = Chroma(
            collection_name="my_first_collection",
            embedding_function=embedding_function,
        )
        self._vector_store.add_documents(documents=all_splits)

    def retrieve_context(self, query):
        """Search internal startup knowledge base for relevant context."""
        docs = self._vector_store.similarity_search(query, k=4)
        return "\n\n".join(d.page_content for d in docs)
