# neurotrace/core/schema.py
"""
Schema definitions for neurotrace core components.
This module defines the data structures used for messages, metadata, and emotion tags.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from neurotrace.core.constants import Role


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
    session_id: Optional[str] = 'default'


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
            "session_id": "default"                # session identifier for context
        }
    }
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)

    def estimated_token_length(self) -> int:
        # todo: Implement a more accurate token counting method
        return self.metadata.token_count or len(self.content.split())

    def to_langchain_message(self) -> Union[HumanMessage, AIMessage]:
        """
        Convert this Message to a LangChain compatible format.
        This is a generic method that can be extended for different message types.
        """
        if self.role == Role.HUMAN:
            return self.to_human_message()
        elif self.role == Role.AI:
            return self.to_ai_message()
        else:
            raise ValueError(f"Unsupported role: {self.role}. Use 'user' or 'ai'.")

    def to_human_message(self) -> HumanMessage:
        """
        Convert this Message to a HumanMessage for LangChain compatibility.
        """
        return HumanMessage(
            id=self.id,
            content=self.content,
            additional_kwargs={
                "id": self.id,
                "metadata": self.metadata.model_dump()
            }
        )

    def to_ai_message(self) -> AIMessage:
        """
        Convert this Message to an AIMessage for LangChain compatibility.
        :return:
        """
        return AIMessage(
            id=self.id,
            content=self.content,
            additional_kwargs={
                "id": self.id,
                "metadata": self.metadata.model_dump()
            }
        )

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False

        # not comparing id
        return (
            self.role == other.role and
            self.content == other.content and
            self.metadata == other.metadata
        )
