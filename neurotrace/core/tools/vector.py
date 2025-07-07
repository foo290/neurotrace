from typing import TYPE_CHECKING

from langchain.tools import Tool

from neurotrace.core.utils import load_prompt

if TYPE_CHECKING:
    from neurotrace.core.vector_memory import BaseVectorMemoryAdapter


def generic_tool_factory(func: callable, tool_name: str, tool_description: str = None, **kwargs) -> Tool:
    """Creates a LangChain Tool with the given function and configuration.

    Args:
        func (callable): The function to be wrapped as a tool.
        tool_name (str): Name of the tool.
        tool_description (str, optional): Description of what the tool does.
            If None, loads description from prompt file. Defaults to None.
        **kwargs: Additional keyword arguments to pass to Tool constructor.

    Returns:
        Tool: A configured LangChain Tool instance.
    """
    return Tool(name=tool_name, func=func, description=tool_description or load_prompt(tool_name), **kwargs)


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
