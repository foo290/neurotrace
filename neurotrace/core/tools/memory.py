from langchain_core.tools import Tool

from neurotrace.core.hippocampus.memory_orchestrator import MemoryOrchestrator
from neurotrace.core.tools.factory import generic_tool_factory
from neurotrace.core.utils import load_prompt


def save_memory_tool(
    memory_orchestrator: MemoryOrchestrator,
    tool_name: str = "save_memory",
    tool_description: str = None,
    **kwargs,
) -> Tool:
    """
    Creates a tool that allows the agent to explicitly save important memories
    to long-term vector memory.

    Args:
        vector_memory_adapter: The adapter to store messages.
        tool_name (str, optional): Name of the tool. Defaults to "save_memory".
        tool_description (str, optional): Description of the tool. Loads from prompt if None.
        **kwargs: Additional keyword args for Tool.

    Returns:
        Tool: A configured LangChain Tool instance for saving memory.
        :param tool_name:
        :param memory_orchestrator:
    """

    def _save(summary: str) -> str:
        """
        Saves a summary in vector memory using the orchestrator.

        Args:
            summary (str): The summary to save.

        Returns:
            str: Confirmation message indicating the summary was saved.
        """
        message_text, convo_tags, *_ = summary.split("-- tags:") + [None, None]
        if convo_tags:
            convo_tags = convo_tags.strip().split(",")
        else:
            convo_tags = []

        message_text = message_text.strip()

        _save_in_vector_memory(memory_orchestrator, message_text, convo_tags)
        _save_in_graph_memory(memory_orchestrator, message_text, convo_tags)
        return "Memory saved in both vector and graph memory."

    return generic_tool_factory(
        func=_save,
        tool_name=tool_name,
        tool_description=tool_description or load_prompt(tool_name),
        **kwargs,
    )


def _save_in_vector_memory(
    memory_orchestrator: MemoryOrchestrator,
    summary: str,
    tags: list[str] = None,
) -> str:
    """
    Saves a summary in vector memory using the orchestrator.

    Args:
        memory_orchestrator (MemoryOrchestrator): The orchestrator managing memory.
        summary (str): The summary to save.
        tags (list[str], optional): Tags associated with the summary. Defaults to None.

    Returns:
        str: Confirmation message indicating the summary was saved.
    """

    memory_orchestrator.save_in_vector_memory(summary, tags=tags)
    return "Vector memory saved."


def _save_in_graph_memory(
    memory_orchestrator: MemoryOrchestrator,
    summary: str,
    tags: list[str] = None,
) -> str:
    """
    Saves a summary in graph memory using the orchestrator.

    Args:
        memory_orchestrator (MemoryOrchestrator): The orchestrator managing memory.
        summary (str): The summary to save.
        tags (list[str], optional): Tags associated with the summary. Defaults to None.

    Returns:
        str: Confirmation message indicating the summary was saved.
    """

    memory_orchestrator.save_in_graph_memory(summary, tags=tags)
    return "Graph memory saved."
