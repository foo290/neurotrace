"""
Vector Database Adapter Module.

This module provides adapter functions for converting between neurotrace's internal
message format and vector database record format. It handles the serialization of
Message objects into a format suitable for vector database storage and retrieval.
"""

from typing import Dict, Any
from neurotrace.core.schema import Message, MessageMetadata


def to_vector_record(msg: Message) -> Dict[str, Any]:
    """
    Converts a Message object to a vector database record format.

    This function transforms a neurotrace Message instance into a dictionary format
    suitable for storage in a vector database. It extracts essential fields including
    the message ID, content text, embedding vector, and all associated metadata.

    Args:
        msg (Message): The Message object to convert.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - id: The message's unique identifier
            - text: The message content
            - embedding: The message's vector embedding
            - metadata: All additional metadata as a dictionary

    Example:
        >>> message = Message(id="123", content="Hello", metadata=MessageMetadata(...))
        >>> record = to_vector_record(message)
        >>> print(record)
        {
            'id': '123',
            'text': 'Hello',
            'embedding': [...],
            'metadata': {...}
        }
    """
    return {
        "id": msg.id,
        "text": msg.content,
        "embedding": msg.metadata.embedding,
        "metadata": msg.metadata.model_dump()
    }
