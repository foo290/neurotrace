"""
Test module for the ShortTermMemory implementation.

This module contains tests that verify the functionality of the ShortTermMemory class,
which manages the temporary storage and retrieval of conversation messages. Tests cover
message append operations, retrieval functionality, and memory management features like
token limits and time-based expiration.
"""

from neurotrace.core.hippocampus.stm import ShortTermMemory
from neurotrace.core.schema import Message, MessageMetadata

from datetime import timedelta, datetime


def test_stm_append_and_retrieve():
    stm = ShortTermMemory(max_tokens=50)
    msg1 = Message(role="user", content="Hello")
    msg2 = Message(role="ai", content="Hi there!")

    stm.append(msg1)
    stm.append(msg2)

    messages = stm.get_messages()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "ai"


def test_stm_token_eviction():
    stm = ShortTermMemory(max_tokens=3)
    stm.append(Message(role="user", content="one two three four"))  # 4 tokens
    stm.append(Message(role="ai", content="ok"))  # 1 token

    # Should evict the first message due to token limit
    messages = stm.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "ai"


def test_stm_clear():
    stm = ShortTermMemory()
    stm.append(Message(role="user", content="Hello"))
    stm.clear()
    assert len(stm.get_messages()) == 0


def test_stm_set_messages():
    msg_list = [
        Message(role="user", content="1"),
        Message(role="ai", content="2")
    ]
    stm = ShortTermMemory(max_tokens=10)
    stm.set_messages(msg_list)
    assert stm.get_messages() == msg_list



# ----------------
# ⚠️ Edge Case Tests
# ----------------

def test_empty_message_content():
    msg = Message(role="user", content="")
    stm = ShortTermMemory(max_tokens=5)
    stm.append(msg)
    assert stm.get_messages()[0].content == ""


def test_missing_token_count_estimates_length():
    msg = Message(role="user", content="This should estimate 5 tokens maybe")
    assert msg.metadata.token_count is None
    assert msg.estimated_token_length() == len(msg.content.split())


def test_extremely_high_token_count_triggers_eviction():
    msg1 = Message(role="user", content="huge!", metadata=MessageMetadata(token_count=999))
    msg2 = Message(role="user", content="huge!", metadata=MessageMetadata(token_count=999))
    stm = ShortTermMemory(max_tokens=10)
    stm.append(msg1)
    stm.append(msg2)
    assert len(stm.get_messages()) == 1


def test_message_without_metadata_defaults_correctly():
    msg = Message(role="ai", content="Auto-meta fallback")
    assert isinstance(msg.metadata, MessageMetadata)
    assert msg.metadata.source == "chat"


def test_message_timestamp_is_recent():
    msg = Message(role="system", content="check timestamp")
    now = datetime.utcnow()
    assert now - msg.timestamp < timedelta(seconds=5)



def test_stm_exact_token_limit_boundary():
    stm = ShortTermMemory(max_tokens=5)

    msg1 = Message(role="user", content="aaa", metadata=MessageMetadata(token_count=2))
    msg2 = Message(role="ai", content="bbb", metadata=MessageMetadata(token_count=3))

    stm.append(msg1)
    stm.append(msg2)

    assert len(stm.get_messages()) == 2  # total = 5, exact

def test_stm_zero_token_limit_clears_all():
    """
    If max_tokens is 0, STM should always be empty.
    """
    stm = ShortTermMemory(max_tokens=0)
    msg = Message(role="user", content="Hi", metadata=MessageMetadata(token_count=0))
    stm.append(msg)

    assert len(stm.get_messages()) == 0


def test_stm_multiple_evictions():
    stm = ShortTermMemory(max_tokens=5)

    stm.append(Message(role="user", content="msg1", metadata=MessageMetadata(token_count=2)))
    stm.append(Message(role="user", content="msg2", metadata=MessageMetadata(token_count=2)))
    stm.append(Message(role="user", content="msg3", metadata=MessageMetadata(token_count=4)))

    msgs = stm.get_messages()
    assert len(msgs) == 1
    assert msgs[0].content == "msg3"

def test_stm_append_overflow_all():
    """
    Should keep the single message even if it exceeds the token budget.
    """
    stm = ShortTermMemory(max_tokens=1)
    msg1 = Message(role="user", content="long message", metadata=MessageMetadata(token_count=3))
    stm.append(msg1)
    assert len(stm.get_messages()) == 1
