from abc import ABC, abstractmethod
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from neurotrace.core.schema import Message


class BaseVectorMemoryAdapter(ABC):
    """Abstract base class for vector memory storage adapters.

    This class defines the interface for storing and retrieving messages using
    vector embeddings. Implementations should provide concrete methods for
    adding, searching, and optionally deleting messages from the vector store.
    """

    @abstractmethod
    def add_messages(self, messages: List[Message]) -> None:
        """Add a list of messages to the vector memory store.

        Args:
            messages (List[Message]): List of Message objects to be added to
                the vector store. Each message will be embedded and stored.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Message]:
        """Search the vector memory for the most relevant messages.

        Performs a similarity search in the vector store using the provided
        query string. The query will be embedded and compared against stored
        message embeddings.

        Args:
            query (str): The search query string.
            k (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            List[Message]: List of messages ranked by similarity to the query,
                limited to k results.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete messages from the vector store by their IDs.

        Args:
            ids (List[str]): List of message IDs to be deleted from the store.

        Note:
            This is an optional operation that might not be supported by all
            vector store implementations.
        """
        pass


class VectorMemoryAdapter(BaseVectorMemoryAdapter):
    """Concrete implementation of vector memory storage using LangChain components.

    This adapter wraps a LangChain-compatible vector store and embedding model
    to provide vector-based message storage and retrieval.

    Args:
        vector_store (VectorStore): LangChain vector store implementation for
            storing embeddings.
        embedding_model (Embeddings): LangChain embeddings model for converting
            text to vectors.
    """

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
        """Add multiple messages to the vector store.

        Converts the messages to LangChain Document format and adds them to
        the underlying vector store. The documents will be automatically
        embedded using the configured embedding model.

        Args:
            messages (List[Message]): List of messages to be added to the
                vector store.
        """
        documents = [msg.to_document() for msg in messages]
        self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 5) -> List[Message]:
        """Search for similar messages in the vector store.

        Performs a similarity search using the query string. The query is
        embedded and compared against stored message embeddings to find
        the most similar messages.

        Args:
            query (str): The search query string.
            k (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            List[Message]: List of messages ranked by similarity to the query,
                limited to k results.

        Note:
            TODO: Add support for enhancing the prompt for vector search using LLM.
        """
        # todo: add support for enhancing the prompt for vector search using llm
        results = self.vector_store.similarity_search(query=query, k=k)
        return [Message.from_document(doc) for doc in results]

    def delete(self, ids: List[str]) -> None:
        """Delete messages from the vector store by their IDs.

        Args:
            ids (List[str]): List of message IDs to be deleted.

        Raises:
            NotImplementedError: If the underlying vector store doesn't
                support deletion operations.
        """
        if hasattr(self.vector_store, "delete"):
            self.vector_store.delete(ids)
        else:
            raise NotImplementedError(f"Delete not supported by {type(self.vector_store)}.")
