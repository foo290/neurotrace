import json
from typing import Any, Dict

from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage

from neurotrace.core.utils import strip_json_code_block
from neurotrace.prompts import task_prompts


def _perform_summarisation(llm: BaseLLM, prompt: PromptTemplate, **kwargs) -> str:
    """
    Perform summarisation using the provided LLM and prompt with dynamic inputs.

    Args:
        llm (BaseLLM): The language model to use for summarisation.
        prompt (PromptTemplate): The prompt with any number of variables.
        **kwargs: The input variables required to fill the prompt.

    Returns:
        str: The summarized or generated output from the LLM.
    """
    formatted_prompt = prompt.format(**kwargs)
    response = llm.invoke(formatted_prompt)
    if isinstance(response, AIMessage):
        response = response.content.strip()
    return response.strip()


def perform_summarisation(llm: BaseLLM, prompt_placeholders: Dict[str, Any], prompt: PromptTemplate = None) -> str:
    """
    Perform summarisation using the provided LLM and prompt with a single message.

    Args:
        llm (BaseLLM): The language model to use for summarisation.
        prompt (PromptTemplate): The prompt with any number of variables.
        message (str): The input text to summarise.

    Returns:
        str: The summarized or generated output from the LLM.
        :param prompt_placeholders:
    """
    prompt = prompt or task_prompts.PROMPT_GENERAL_SUMMARY
    return _perform_summarisation(llm=llm, prompt=prompt, **prompt_placeholders)


def get_graph_summary(llm: BaseLLM, text: str) -> str:
    """
    Get a graph summary from the LLM.

    Args:
        llm (BaseLLM): The language model to use for summarisation.
        text (str): The input text to summarize.

    Returns:
        str: The graph summary generated by the LLM.
    """
    response = _perform_summarisation(llm=llm, prompt=task_prompts.PROMPT_GRAPH_SUMMARY, message=text)
    response = strip_json_code_block(response)
    return response


def get_vector_and_graph_summary(llm: BaseLLM, text: str) -> Dict[str, str]:
    """
    Get vector and graph summaries from the LLM.

    Args:
        llm (BaseLLM): The language model to use for summarisation.
        text (str): The input text to summarize.

    Returns:
        Tuple[str, str]: A tuple containing the vector summary and graph summary.
    """
    response = _perform_summarisation(llm=llm, prompt=task_prompts.PROMPT_VECTOR_AND_GRAPH_SUMMARY, message=text)
    response = strip_json_code_block(response)
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        return {}
