# tests/conftest.py
import pytest
from neurotrace.core.schema import Message, MessageMetadata

@pytest.fixture
def sample_message():
    return Message(role="user", content="Hello!", metadata=MessageMetadata(tags=["sample"]))
