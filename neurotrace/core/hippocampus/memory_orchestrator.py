from typing import Any, Dict, List, Union

from langchain.llms.base import BaseLLM
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel

from neurotrace.core.constants import Role
from neurotrace.core.graph_memory import GraphMemoryAdapter, GraphTripletIndexer
from neurotrace.core.schema import Message, MessageMetadata
from neurotrace.core.vector_memory import VectorMemoryAdapter


class MemoryOrchestrator:
    """Manages both short-term and long-term memory for Neurotrace agents."""

    def __init__(
        self,
        llm: Union[BaseLLM, BaseChatModel],
        graph_store: GraphStore,
        vector_store: VectorStore,
    ):
        self.llm = llm
        self._graph_indexer = GraphTripletIndexer(llm)
        self._graph_memory_adapter = GraphMemoryAdapter(llm, graph_store, self._graph_indexer)
        self._vector_memory_adapter = VectorMemoryAdapter(vector_store)

    def save_in_graph_memory(self, summary: str, tags: List[str] = None) -> str:
        """Saves a summary in graph memory."""
        # todo: interface this with Message class
        self._graph_memory_adapter.add_conversation(summary, tags=tags)
        return "Graph memory saved."

    def save_in_vector_memory(self, summary: str, tags: List[str] = None) -> str:
        """Saves a summary in vector memory."""
        message = Message(role=Role.AI.value, content=summary, metadata=MessageMetadata(tags=tags))
        self._vector_memory_adapter.add_messages([message])
        return "Vector memory saved."

    def search_vector_memory(self, query: str, k: int = 5) -> List[Message]:
        return self._vector_memory_adapter.search(query, k)

    def search_graph_memory(self, query: str) -> str:
        return self._graph_memory_adapter.ask_graph(query)["result"]
