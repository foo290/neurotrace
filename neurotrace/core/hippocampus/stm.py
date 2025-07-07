# neurotrace/core/_stm.py
import uuid
from abc import ABC, abstractmethod
from typing import List

from neurotrace.core.schema import Message


class BaseShortTermMemory(ABC):
    """Abstract base class for short-term memory management.

    This class defines the interface for managing a temporary message store
    with token limit constraints.
    """

    @abstractmethod
    def append(self, message: Message) -> None:
        """Add a message to short-term memory.

        Args:
            message (Message): The message to be added to memory.
        """
        ...

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Retrieve all messages from short-term memory.

        Returns:
            List[Message]: List of all messages currently in memory.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from short-term memory."""
        ...

    @abstractmethod
    def set_messages(self, messages: List[Message]) -> None:
        """Replace all messages in memory with a new list.

        Args:
            messages (List[Message]): New list of messages to store.
        """
        ...

    @abstractmethod
    def total_tokens(self) -> int:
        """Calculate total tokens used by all messages.

        Returns:
            int: Sum of estimated token lengths of all messages.
        """
        ...

    def __len__(self) -> int:
        """Get the number of messages in memory.

        Returns:
            int: Count of messages currently stored.
        """
        return len(self.get_messages())


class ShortTermMemory(BaseShortTermMemory):
    """Implementation of token-limited short-term memory.

    This class maintains a list of messages while ensuring the total token
    count stays within a specified limit. When the limit is exceeded, older
    messages are automatically evicted.

    Args:
        max_tokens (int, optional): Maximum number of tokens to store. Set to 0
            to disable memory (all messages will be evicted). Defaults to 2048.
    """

    def __init__(self, max_tokens: int = 2048):
        """Initialize short-term memory with token limit.

        Args:
            max_tokens (int, optional): Maximum number of tokens to store.
                Defaults to 2048.
        """
        self.messages: List[Message] = []
        self.max_tokens = max_tokens

    def append(self, message: Message) -> None:
        """Add a message to memory and evict old messages if needed.

        If the message doesn't have an ID, generates a UUID for it. After
        adding the message, ensures token limit compliance by evicting old
        messages if necessary.

        Args:
            message (Message): The message to add to memory.
        """
        if not message.id:
            message.id = str(uuid.uuid4())

        self.messages.append(message)
        self._evict_if_needed()

    def get_messages(self) -> List[Message]:
        """Get all messages currently in memory.

        Returns:
            List[Message]: List of all stored messages.
        """
        return self.messages

    def clear(self) -> None:
        """Remove all messages from memory."""
        self.messages = []

    def _evict_if_needed(self) -> None:
        """Maintain token limit by removing oldest messages.

        If max_tokens is 0, clears all messages. Otherwise, removes oldest
        messages until total token count is within limit, always keeping at
        least one message.
        """
        # If max_tokens is 0, clear everything (user wants no memory)
        if self.max_tokens == 0:
            self.messages.clear()
            return

        total = sum(msg.estimated_token_length() for msg in self.messages)

        # Keep at least 1 message even if over limit (unless max_tokens is zero)
        while total > self.max_tokens and len(self.messages) > 1:
            total -= self.messages[0].estimated_token_length()
            self.messages.pop(0)

    def set_messages(self, messages: List[Message]) -> None:
        """Replace current messages with new list and maintain token limit.

        Args:
            messages (List[Message]): New messages to store in memory.
        """
        self.messages = messages
        self._evict_if_needed()

    def total_tokens(self) -> int:
        """Calculate total tokens used by all messages.

        Returns:
            int: Sum of estimated token lengths across all messages.
        """
        return sum(m.estimated_token_length() for m in self.messages)

    def __len__(self):
        """Get number of messages in memory.

        Returns:
            int: Count of stored messages.
        """
        return len(self.messages)

    def __repr__(self):
        """Get string representation of memory state.

        Returns:
            str: String showing message count and token usage/limit.
        """
        return f"<STM messages={len(self.messages)} tokens={self.total_tokens()}/{self.max_tokens}>"
