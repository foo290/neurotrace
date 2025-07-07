# neurotrace/core/schema.py
"""
Schema definitions for neurotrace core components.

This module defines the data structures used for managing neuromorphic memory components
including messages, metadata, and emotion tags using Pydantic models.

Note:
    All models inherit from Pydantic BaseModel for data validation.
"""

import uuid
from datetime import UTC, datetime
from typing import List, Literal, Optional, Union

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from neurotrace.core.constants import Role


class EmotionTag(BaseModel):
    """Represents the emotional context and intensity of a message.

    Attributes:
        sentiment (Optional[Literal["positive", "neutral", "negative"]]): The emotional
            tone of the message. Defaults to None.
        intensity (Optional[float]): A value indicating the strength of the emotion.
            Defaults to None.
    """

    sentiment: Optional[Literal["positive", "neutral", "negative"]] = None
    intensity: Optional[float] = None


class MessageMetadata(BaseModel):
    """Contains additional contextual information about a message.

    Attributes:
        token_count (Optional[int]): Number of tokens in the associated message.
        embedding (Optional[List[float]]): Vector representation of the message content.
        source (Optional[Literal["chat", "web", "api", "system"]]): Origin of the message.
            Defaults to "chat".
        tags (Optional[List[str]]): List of categorical tags. Defaults to empty list.
        thread_id (Optional[str]): Unique identifier for the conversation thread.
        user_id (Optional[str]): Identifier for the message author.
        related_ids (Optional[List[str]]): References to related message IDs.
        emotions (Optional[EmotionTag]): Emotional analysis of the message.
        compressed (Optional[bool]): Whether the message content is compressed.
            Defaults to False.
        session_id (Optional[str]): Current session identifier. Defaults to 'default'.
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
    session_id: Optional[str] = "default"


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)

    def estimated_token_length(self) -> int:
        """Estimates the number of tokens in the message content.

        Returns:
            int: The estimated token count, either from metadata or word-based count.

        Note:
            Currently uses a simple word-splitting approach if token_count is not set
            in metadata. TODO: Implement a more accurate token counting method.
        """
        # todo: Implement a more accurate token counting method
        return self.metadata.token_count or len(self.content.split())

    def to_langchain_message(self) -> Union[HumanMessage, AIMessage]:
        """Converts this Message to a LangChain compatible format.

        Returns:
            Union[HumanMessage, AIMessage]: A LangChain message object based on the role.

        Raises:
            ValueError: If the role is neither 'human' nor 'ai'.
        """
        if Role.from_string(self.role) is Role.HUMAN:
            return self.to_human_message()
        elif Role.from_string(self.role) is Role.AI:
            return self.to_ai_message()
        else:
            raise ValueError(f"Unsupported role: {self.role}. Use 'human' or 'ai'.")

    def to_human_message(self) -> HumanMessage:
        """Converts this Message to a LangChain HumanMessage format.

        Returns:
            HumanMessage: A LangChain HumanMessage with the message content and metadata.
        """
        return HumanMessage(
            id=self.id, content=self.content, additional_kwargs={"id": self.id, "metadata": self.metadata.model_dump()}
        )

    def to_ai_message(self) -> AIMessage:
        """Converts this Message to a LangChain AIMessage format.

        Returns:
            AIMessage: A LangChain AIMessage with the message content and metadata.
        """
        return AIMessage(
            id=self.id, content=self.content, additional_kwargs={"id": self.id, "metadata": self.metadata.model_dump()}
        )

    def to_document(self) -> Document:
        """
        Convert Message to LangChain-compatible Document with safe metadata.
        """
        raw_metadata = self.metadata.model_dump() if self.metadata else {}
        doc = Document(page_content=self.content, metadata={"id": self.id, "role": self.role, **raw_metadata})
        doc = filter_complex_metadata([doc])  # Remove lists, dicts, etc.

        return doc[0]

    @staticmethod
    def from_document(doc: Document) -> "Message":
        metadata = doc.metadata or {}
        role_str = metadata.pop("role", Role.HUMAN.value)  # fallback to human
        return Message(
            role=Role.from_string(role_str),
            content=doc.page_content,
            metadata=MessageMetadata(**metadata) if metadata else None,
            id=metadata.get("id"),
        )

    def __eq__(self, other):
        """Checks equality between two Message objects.

        Args:
            other: Another object to compare with.

        Returns:
            bool: True if the messages have the same role, content, and metadata
                (excluding id), False otherwise.
        """
        if not isinstance(other, Message):
            return False

        # not comparing id
        return self.role == other.role and self.content == other.content and self.metadata == other.metadata
