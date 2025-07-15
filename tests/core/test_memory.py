"""
Test module for the NeurotraceMemory class.

This module contains tests to verify the functionality of the NeurotraceMemory class,
which implements a custom memory system for LangChain integrations. Tests cover memory
variable handling, context preservation, and message conversion functionality.
"""

from unittest.mock import MagicMock

import pytest
from langchain.schema import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel

from neurotrace.core.constants import Role
from neurotrace.core.memory import NeurotraceMemory
from neurotrace.core.schema import Message, MessageMetadata


@pytest.fixture
def mock_llm():
    return MagicMock(spec=BaseChatModel)


def test_memory_variables_returns_expected_key(mock_llm):
    memory = NeurotraceMemory(mock_llm)
    assert memory.memory_variables == ["chat_history"]


def test_save_and_load_context_preserves_content(mock_llm):
    memory = NeurotraceMemory(max_tokens=20, llm=mock_llm)

    inputs = {"input": "What's your name?"}
    outputs = {"output": "I'm Neurotrace."}

    memory.save_context(inputs, outputs)

    history = memory.load_memory_variables({})
    msgs = history["chat_history"]
    print(msgs)

    assert isinstance(msgs, list)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "What's your name?"
    assert msgs[1].content == "I'm Neurotrace."


def test_message_format_conversion_matches_neurotrace_model(mock_llm):
    memory = NeurotraceMemory(mock_llm)

    memory.save_context({"input": "Hi"}, {"output": "Hello"})

    # Internally, should have 2 messages in custom format
    internal_msgs = memory._stm.get_messages()
    assert len(internal_msgs) == 2

    user_msg = internal_msgs[0]
    ai_msg = internal_msgs[1]

    assert user_msg.role == "human"
    assert ai_msg.role == "ai"
    assert user_msg.content == "Hi"
    assert ai_msg.content == "Hello"
    assert user_msg.metadata is not None
    assert user_msg.metadata.source == "chat"


def test_conversion_preserves_metadata_fields(mock_llm):
    memory = NeurotraceMemory(mock_llm)
    msg = Message(
        role="human",
        content="Check metadata",
        metadata=MessageMetadata(token_count=5, tags=["test", "meta"], source="chat"),
    )
    memory._stm.append(msg)

    # Converted back to LangChain format
    loaded = memory.load_memory_variables({})
    assert isinstance(loaded["chat_history"][0], HumanMessage)
    assert loaded["chat_history"][0].content == "Check metadata"


def test_memory_clear_empties_internal_stm(mock_llm):
    memory = NeurotraceMemory(mock_llm)
    memory.save_context({"input": "To be deleted"}, {"output": "Gone"})

    assert len(memory._stm.get_messages()) == 2
    memory.clear()
    assert len(memory._stm.get_messages()) == 0


def test_save_context_handles_missing_input_output_keys(mock_llm):
    memory = NeurotraceMemory(mock_llm)
    memory.save_context({}, {})  # no "input" or "output" keys

    msgs = memory._stm.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == ""
    assert msgs[1].content == ""


def test_eviction_behavior_under_token_budget(mock_llm):
    memory = NeurotraceMemory(max_tokens=5, llm=mock_llm)

    memory._stm.append(Message(role="user", content="first", metadata=MessageMetadata(token_count=3)))
    memory._stm.append(Message(role="ai", content="second", metadata=MessageMetadata(token_count=3)))

    # Should evict the first message
    msgs = memory.load_memory_variables({})["chat_history"]
    assert len(msgs) == 1
    assert msgs[0].content == "second"


def test_conversion_round_trip_integrity(mock_llm):
    memory = NeurotraceMemory(mock_llm)

    memory.save_context({"input": "Ping"}, {"output": "Pong"})

    # LangChain → Message → LangChain
    loaded_msgs = memory.load_memory_variables({})
    assert isinstance(loaded_msgs["chat_history"][0], HumanMessage)
    assert loaded_msgs["chat_history"][0].content == "Ping"


def test_neurotrace_memory_with_ltm_and_stm(mock_llm):
    # Setup
    history = InMemoryChatMessageHistory()
    session_id = "default"
    memory = NeurotraceMemory(max_tokens=100, history=history, session_id=session_id, llm=mock_llm)

    # Define expected messages
    user_msg = Message(
        id="boom",
        role=Role.HUMAN.value,
        content="What's the weather like today?",
        metadata=MessageMetadata(session_id=session_id),
    )
    ai_msg = Message(
        id="boom",
        role=Role.AI.value,
        content="It's sunny and warm!",
        metadata=MessageMetadata(session_id=session_id),
    )

    # Act
    memory.save_context(inputs={"input": user_msg.content}, outputs={"output": ai_msg.content})

    # Assert STM
    stm_msgs = memory._stm.get_messages()
    assert stm_msgs == [user_msg, ai_msg]

    # Assert LTM
    ltm_msgs = memory._ltm.get_messages()
    assert ltm_msgs == [user_msg, ai_msg]

    # Assert underlying LangChain messages for sanity
    lc_msgs = history.messages
    assert isinstance(lc_msgs[0], HumanMessage)
    assert lc_msgs[0].content == user_msg.content
    assert isinstance(lc_msgs[1], AIMessage)
    assert lc_msgs[1].content == ai_msg.content
