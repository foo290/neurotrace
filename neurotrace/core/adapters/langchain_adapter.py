"""
LangChain Adapter Module.

This module provides adapter functions for converting between neurotrace's internal
Message format and LangChain's message formats. It handles bidirectional conversion
between neurotrace Messages and LangChain's HumanMessage/AIMessage types.
"""

from typing import List, Optional, cast, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage

from neurotrace.core.schema import Message


def from_langchain_message(msg: BaseMessage, role: Optional[str] = None) -> Message:
    """
    Convert a LangChain message to a neurotrace Message.

    This function transforms a LangChain message into neurotrace's Message format,
    with automatic role detection based on the message type or an optional
    explicit role override.

    Args:
        msg (BaseMessage): The LangChain message to convert.
        role (Optional[str], optional): Explicitly specify the role to use.
            If None, role is detected from the message type. Defaults to None.

    Returns:
        Message: A neurotrace Message with:
            - role determined by message type or override
            - content from the original message

    Example:
        >>> lc_msg = HumanMessage(content="Hello")
        >>> msg = from_langchain_message(lc_msg)
        >>> print(msg.role)  # "user"
        >>> print(msg.content)  # "Hello"
    """
    detected_role = (
        role if role else
        "human" if isinstance(msg, HumanMessage) else
        "ai" if isinstance(msg, AIMessage) else
        "system"
    )

    role_literal = cast(Literal["user", "ai", "system"], detected_role)
    return Message(
        role=role_literal,
        content=msg.content
    )