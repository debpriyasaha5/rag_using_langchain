from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os

load_dotenv()
chunks = 1000
embedding_model = "text-embedding-3-small"
expected_dimension = 1536


class Rag_Core:
    def indexing(self):
        pdf_path = Path(__file__).with_name("Dsa.pdf")
        loader = PyPDFLoader(str(pdf_path))
        raw_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunks, chunk_overlap=0)
        return text_splitter.split_documents(raw_docs)


    def create_embeddings(self):
        embeddings = OpenAIEmbeddings(model=embedding_model)
        print(f"Using OpenAI embedding model: {embedding_model}")
        return embeddings

    def load_vector_store(self, embeddings):
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set in the environment.")

        client = self.get_pinecone_client()
        if not client.has_index(index_name):
            raise ValueError(
                f"Pinecone index '{index_name}' does not exist yet. Run the indexing step once before chatting."
            )

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        print(f"Loaded existing Pinecone index '{index_name}'.")
        return vector_store

    def get_pinecone_client(self):
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        if not api_key or not environment:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in the environment.")
        return Pinecone(api_key=api_key, environment=environment)


    def ensure_pinecone_index(self, client, index_name, dimension):
        if client.has_index(index_name):
            info = client.describe_index(index_name)
            idx_dim = info.get("dimension") if isinstance(info, dict) else getattr(info, "dimension", None)
            if idx_dim != dimension:
                raise ValueError(
                    f"Pinecone index '{index_name}' exists with dimension {idx_dim}, "
                    f"but the embedding model produces vectors of dimension {dimension}. "
                    "Use an index with the matching dimension or update PINECONE_INDEX_NAME."
                )
            print(f"Pinecone index '{index_name}' already exists with dimension {dimension}.")
        else:
            spec = {"serverless": {"cloud": "aws", "region": os.getenv("PINECONE_ENVIRONMENT")}}
            client.create_index(name=index_name, dimension=dimension, metric="cosine", spec=spec)
            print(f"Created Pinecone index '{index_name}' with dimension {dimension}.")


    def store_vectors_in_pinecone(self, embeddings):
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set in the environment.")

        client = self.get_pinecone_client()
        self.ensure_pinecone_index(client, index_name, expected_dimension)

        vector_store = PineconeVectorStore.from_documents(
            documents=self.indexing(),
            embedding=embeddings,
            index_name=index_name,
        )
        print("Vectors stored in Pinecone index.")
        return vector_store


# if __name__ == "__main__":
#     embeddings = create_embeddings()
#     store_vectors_in_pinecone(embeddings)
