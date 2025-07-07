from typing import TYPE_CHECKING

from langchain.tools import Tool

from neurotrace.core.utils import load_prompt

if TYPE_CHECKING:
    from neurotrace.core.vector_memory import BaseVectorMemoryAdapter


def generic_tool_factory(func: callable, tool_name: str, tool_description: str = None, **kwargs) -> Tool:
    return Tool(name=tool_name, func=func, description=tool_description or load_prompt(tool_name), **kwargs)


def vector_memory_search_tool(
    vector_memory_adapter: "BaseVectorMemoryAdapter",
    tool_name: str = "search_memory",
    tool_description: str = None,
    **kwargs,
) -> Tool:
    """
    Function to create a search tool for vector memory.
    This tool will use the `search` method of the provided vector memory adapter.

    :param vector_memory_adapter:
    :param tool_name:
    :param tool_description:
    :param kwargs:
    :return:
    """
    return generic_tool_factory(
        func=vector_memory_adapter.search,
        tool_name=tool_name,
        tool_description=tool_description or load_prompt(tool_name),
        **kwargs,
    )
