"""
Test module for vector database adapter functionality.

This module verifies the conversion of Message objects to vector database records,
ensuring proper handling of message content, embeddings, and metadata during the
transformation process for vector similarity search operations.
"""

from neurotrace.core.schema import Message, MessageMetadata
from neurotrace.core.adapters.vector_db_adapter import to_vector_record


def test_to_vector_record_formatting():
    msg = Message(
        role="user",
        content="Vector me!",
        metadata=MessageMetadata(
            embedding=[0.1, 0.2, 0.3],
            tags=["test"]
        )
    )

    record = to_vector_record(msg)

    assert record["id"] == msg.id
    assert record["text"] == "Vector me!"
    assert isinstance(record["embedding"], list)
    assert record["metadata"]["tags"] == ["test"]
