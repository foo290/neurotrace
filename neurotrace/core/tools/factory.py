from langchain.tools import Tool

from neurotrace.core.utils import load_prompt


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
