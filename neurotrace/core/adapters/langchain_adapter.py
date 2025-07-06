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


def to_langchain_messages(messages: List["Message"]) -> List[HumanMessage | AIMessage]:
    """
    Convert a list of neurotrace Messages to LangChain message format.

    This function transforms neurotrace Message objects into their corresponding
    LangChain message types based on their roles. Currently supports conversion
    to HumanMessage and AIMessage types.

    Args:
        messages (List[Message]): A list of neurotrace Message objects to convert.

    Returns:
        List[HumanMessage | AIMessage]: A list of LangChain message objects where:
            - Messages with role="user" become HumanMessage
            - Messages with role="ai" become AIMessage

    Example:
        >>> msgs = [Message(role="user", content="Hello"), Message(role="ai", content="Hi")]
        >>> langchain_msgs = to_langchain_messages(msgs)
        >>> print(isinstance(langchain_msgs[0], HumanMessage))  # True
    """
    converted = []
    for msg in messages:
        if msg.role == "user":
            converted.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            converted.append(AIMessage(content=msg.content))
    return converted


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
        "user" if isinstance(msg, HumanMessage) else
        "ai" if isinstance(msg, AIMessage) else
        "system"
    )

    role_literal = cast(Literal["user", "ai", "system"], detected_role)
    return Message(
        role=role_literal,
        content=msg.content
    )