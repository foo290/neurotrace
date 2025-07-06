"""
Test module for LangChain adapter functionality.

This module contains tests that verify the bidirectional conversion between Neurotrace
Message objects and LangChain message types, ensuring proper preservation of message
content, roles, and metadata during conversion.
"""

from neurotrace.core.schema import Message
from neurotrace.core.adapters.langchain_adapter import from_langchain_message
from langchain_core.messages import HumanMessage, AIMessage


def test_to_langchain_messages_conversion():
    msgs = [
        Message(role="human", content="Hi"),
        Message(role="ai", content="Hello!")
    ]
    converted = [i.to_langchain_message() for i in msgs]

    assert isinstance(converted[0], HumanMessage)
    assert isinstance(converted[1], AIMessage)
    assert converted[0].content == "Hi"
    assert converted[1].content == "Hello!"


def test_from_langchain_message():
    lc_msg = HumanMessage(content="Hey there!")
    msg = from_langchain_message(lc_msg)

    assert msg.role == "human"
    assert msg.content == "Hey there!"
