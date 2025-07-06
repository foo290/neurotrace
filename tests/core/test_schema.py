from neurotrace.core.schema import Message, MessageMetadata, EmotionTag
from datetime import datetime


def test_message_creation():
    msg = Message(
        role="user",
        content="Hello!",
        metadata=MessageMetadata(
            token_count=5,
            emotions=EmotionTag(sentiment="positive", intensity=0.8)
        )
    )

    assert msg.role == "user"
    assert msg.content == "Hello!"
    assert msg.metadata.token_count == 5
    assert isinstance(msg.timestamp, datetime)
    assert msg.metadata.emotions.sentiment == "positive"


def test_token_estimation_fallback():
    msg = Message(role="user", content="Hello world")
    assert msg.estimated_token_length() == 2
