from typing import TYPE_CHECKING

from langchain.tools import Tool

from neurotrace.core.constants import Role
from neurotrace.core.schema import Message
from neurotrace.core.tools.factory import generic_tool_factory
from neurotrace.core.utils import load_prompt

if TYPE_CHECKING:
    from neurotrace.core.vector_memory import BaseVectorMemoryAdapter


def vector_memory_search_tool(
    vector_memory_adapter: "BaseVectorMemoryAdapter",
    tool_name: str = "search_memory",
    tool_description: str = None,
    **kwargs,
) -> Tool:
    """Creates a search tool for vector memory using the provided adapter.

    This factory function creates a LangChain Tool that wraps the search functionality
    of a vector memory adapter. The tool can be used to perform similarity searches
    in the vector store.

    Args:
        vector_memory_adapter (BaseVectorMemoryAdapter): The adapter instance that
            provides vector memory search functionality.
        tool_name (str, optional): Name of the search tool. Defaults to "search_memory".
        tool_description (str, optional): Description of what the search tool does.
            If None, loads description from prompt file. Defaults to None.
        **kwargs: Additional keyword arguments to pass to Tool constructor.

    Returns:
        Tool: A configured LangChain Tool instance for vector memory search.
    """
    return generic_tool_factory(
        func=vector_memory_adapter.search,
        tool_name=tool_name,
        tool_description=tool_description or load_prompt(tool_name),
        **kwargs,
    )


def save_memory(
    vector_memory_adapter: "BaseVectorMemoryAdapter",
    tool_name: str = "save_memory",
    tool_description: str = None,
    **kwargs,
) -> Tool:
    """
    Creates a tool that allows the agent to explicitly save important memories
    to long-term vector memory.

    Args:
        vector_memory_adapter (BaseVectorMemoryAdapter): The adapter to store messages.
        tool_name (str, optional): Name of the tool. Defaults to "save_memory".
        tool_description (str, optional): Description of the tool. Loads from prompt if None.
        **kwargs: Additional keyword args for Tool.

    Returns:
        Tool: A LangChain-compatible tool.
    """

    def _save_memory(summary: str) -> str:
        message = Message(role=Role.HUMAN.value, content=summary)
        vector_memory_adapter.add_messages([message])
        return "Memory saved."

    return generic_tool_factory(
        func=_save_memory,
        tool_name=tool_name,
        tool_description=tool_description or load_prompt(tool_name),
        **kwargs,
    )
