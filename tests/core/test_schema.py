import pytest
from neurotrace.core.schema import Message, MessageMetadata, EmotionTag
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from uuid import UUID


def test_message_creation_with_defaults():
    msg = Message(role="user", content="Hello, world!")

    # UUID valid
    UUID(msg.id)  # Will raise ValueError if not valid

    assert msg.role == "user"
    assert msg.content == "Hello, world!"
    assert isinstance(msg.timestamp, datetime)

    # Metadata defaults
    assert isinstance(msg.metadata, MessageMetadata)
    assert msg.metadata.source == "chat"
    assert msg.metadata.tags == []
    assert msg.metadata.session_id == "default"
    assert msg.metadata.compressed is False


def test_estimated_token_length_with_explicit_token_count():
    metadata = MessageMetadata(token_count=42)
    msg = Message(role="ai", content="ignored", metadata=metadata)
    assert msg.estimated_token_length() == 42


def test_estimated_token_length_with_no_token_count():
    msg = Message(role="ai", content="This is a test message.")
    assert msg.estimated_token_length() == 5  # 5 words


def test_to_human_message_conversion():
    msg = Message(role="user", content="Hey!")
    human_msg = msg.to_human_message()

    assert isinstance(human_msg, HumanMessage)
    assert human_msg.content == msg.content
    assert human_msg.additional_kwargs["metadata"] == msg.metadata.model_dump()


def test_to_ai_message_conversion():
    msg = Message(role="ai", content="Hello there!")
    ai_msg = msg.to_ai_message()

    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == msg.content
    assert ai_msg.additional_kwargs["metadata"] == msg.metadata.model_dump()


def test_emotion_tag_structure():
    emotion = EmotionTag(sentiment="positive", intensity=0.9)
    assert emotion.sentiment == "positive"
    assert 0.0 <= emotion.intensity <= 1.0


def test_partial_metadata_support():
    metadata = MessageMetadata(tags=["urgent", "project-x"], user_id="nitin")
    msg = Message(role="user", content="Edge case test", metadata=metadata)

    assert msg.metadata.tags == ["urgent", "project-x"]
    assert msg.metadata.user_id == "nitin"
    assert msg.metadata.source == "chat"  # default


def test_empty_content_should_work():
    msg = Message(role="system", content="")
    assert msg.content == ""
    assert msg.estimated_token_length() == 0


def test_invalid_sentiment_should_fail():
    with pytest.raises(ValueError):
        EmotionTag(sentiment="excited")  # Invalid literal


def test_uuid_is_unique():
    m1 = Message(role="user", content="One")
    m2 = Message(role="user", content="Two")
    assert m1.id != m2.id
