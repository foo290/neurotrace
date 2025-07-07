from abc import ABC, abstractmethod
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from neurotrace.core.schema import Message


class BaseVectorMemoryAdapter(ABC):
    @abstractmethod
    def add_messages(self, messages: List[Message]) -> None:
        """
        Add a list of messages to the vector memory store.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Message]:
        """
        Search the vector memory for the most relevant messages.
        Returns a list of messages ranked by similarity.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Optional: delete messages based on their unique IDs.
        """
        pass


class VectorMemoryAdapter(BaseVectorMemoryAdapter):
    def __init__(self, vector_store: VectorStore, embedding_model: Embeddings):
        """
        Vector memory adapter that wraps a LangChain-compatible vector store.

        Args:
            vector_store (VectorStore): Any LangChain-compatible vector store.
            embedding_model (Embeddings): Embedding model to generate embeddings.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def add_messages(self, messages: List[Message]) -> None:
        documents = [msg.to_document() for msg in messages]
        self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 5) -> List[Message]:
        # todo: add support for enhancing the prompt for vector search using llm
        results = self.vector_store.similarity_search(query=query, k=k)
        return [Message.from_document(doc) for doc in results]

    def delete(self, ids: List[str]) -> None:
        if hasattr(self.vector_store, "delete"):
            self.vector_store.delete(ids)
        else:
            raise NotImplementedError(f"Delete not supported by {type(self.vector_store)}.")
