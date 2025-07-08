# neurotrace/core/ltm.py

from abc import ABC, abstractmethod
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from neurotrace.core.adapters.langchain_adapter import from_langchain_message
from neurotrace.core.constants import Role
from neurotrace.core.schema import Message


class BaseLongTermMemory(ABC):
    """Abstract base class for long-term memory storage.

    This class defines the interface for persistent storage of conversation
    messages. Implementations should handle the storage and retrieval of
    messages across different chat sessions.
    """

    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Store a message in long-term memory.

        Args:
            message (Message): The message object to be stored.
        """
        pass

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """Add a user message to long-term memory.

        Args:
            content (str): The content of the user's message.
        """
        pass

    @abstractmethod
    def add_ai_message(self, content: str) -> None:
        """Add an AI message to long-term memory.

        Args:
            content (str): The content of the AI's message.
        """
        pass

    @abstractmethod
    def get_messages(self, session_id: str) -> List[Message]:
        """Retrieve all messages for a given session.

        Args:
            session_id (str): The identifier for the chat session.

        Returns:
            List[Message]: List of messages associated with the session.
        """
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Clear messages for a given session.

        Args:
            session_id (str): The identifier for the chat session to clear.
        """
        pass


class LongTermMemory(BaseLongTermMemory):
    """LangChain chat history adapter for long-term memory storage.

    This adapter implements the BaseLongTermMemory interface using LangChain's
    chat history components for persistent storage.

    Args:
        history (BaseChatMessageHistory): LangChain chat history implementation
            to use for storage.
        session_id (str, optional): Default session identifier. Defaults to "default".
    """

    def __init__(self, history: BaseChatMessageHistory, session_id: str = "default"):
        """Initialize the LangChain history adapter.

        Args:
            history (BaseChatMessageHistory): LangChain chat history implementation.
            session_id (str, optional): Default session identifier. Defaults to "default".
        """
        self.history = history
        self.session_id = session_id

    def add_message(self, message: Message) -> None:
        """Add a message to the LangChain chat history.

        Converts the Message object to LangChain's message format before storing.

        Args:
            message (Message): The message to store.
        """
        lc_msg: BaseMessage = message.to_langchain_message()
        self.history.add_message(lc_msg)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the chat history.

        Creates a Message object with HUMAN role and adds it to storage.

        Args:
            content (str): The content of the user's message.
        """
        self.add_message(Message(role=Role.HUMAN.value, content=content))

    def add_ai_message(self, content: str) -> None:
        """Add an AI message to the chat history.

        Creates a Message object with AI role and adds it to storage.

        Args:
            content (str): The content of the AI's message.
        """
        self.add_message(Message(role=Role.AI.value, content=content))

    def get_messages(self, session_id: str = None) -> List[Message]:
        """Retrieve all messages from the chat history.

        Args:
            session_id (str, optional): Session identifier. Currently unused as
                LangChain history doesn't support session filtering. Defaults to None.

        Returns:
            List[Message]: All messages in the chat history.
        """
        lc_msgs = self.history.messages
        return [from_langchain_message(m) for m in lc_msgs]

    def clear(self, session_id: str = None) -> None:
        """Clear all messages from the chat history.

        Args:
            session_id (str, optional): Session identifier. Currently unused as
                LangChain history doesn't support session filtering. Defaults to None.
        """
        self.history.clear()
