# neurotrace/core/schema.py
"""
Schema definitions for neurotrace core components.
This module defines the data structures used for messages, metadata, and emotion tags.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage


class EmotionTag(BaseModel):
    """
    EmotionTag represents the emotional context of a message.
    """
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = None
    intensity: Optional[float] = None


class MessageMetadata(BaseModel):
    """
    MessageMetadata contains additional information about a message.
    """
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    source: Optional[Literal["chat", "web", "api", "system"]] = "chat"
    tags: Optional[List[str]] = []
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    related_ids: Optional[List[str]] = []
    emotions: Optional[EmotionTag] = None
    compressed: Optional[bool] = False


class Message(BaseModel):
    """
    Message represents a single communication in the system.
    It includes the sender's role, content, timestamp, and metadata.
    Each message has a unique identifier generated as a UUID.

    Example Representation:
    {
        "id": "<uuid4 or hash>",                    # unique message ID
        "role": "user" | "ai" | "system",           # sender role
        "content": "string",                        # message text
        "timestamp": "ISO 8601",                    # message time (UTC)
        "metadata": {
            "token_count": 32,                      # optional, for budgeting/compression
            "embedding": [...],                     # vector representation (optional in-memory or precomputed)
            "source": "chat" | "web" | "api",       # source of message
            "tags": ["finance", "personal"],        # custom tags for search
            "thread_id": "conversation_XYZ",        # optional thread/conversation tracking
            "user_id": "abc123",                    # to associate memory across sessions
            "related_ids": ["msg_id_1", "msg_id_2"],# links to other related messages (graph edge)
            "emotions": {"sentiment": "positive", "intensity": 0.85},  # optional emotion tagging
            "compressed": False                     # for summarization/compression tracking
        }
    }
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "ai", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)

    def estimated_token_length(self) -> int:
        # todo: Implement a more accurate token counting method
        return self.metadata.token_count or len(self.content.split())
