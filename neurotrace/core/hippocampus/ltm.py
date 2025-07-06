# neurotrace/core/ltm.py

from abc import ABC, abstractmethod

from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from neurotrace.core.adapters.langchain_adapter import from_langchain_message

from neurotrace.core.schema import Message
from neurotrace.core.constants import Role



class LongTermMemory(ABC):
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """
        Store a message in long-term memory.
        """
        pass

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        pass

    @abstractmethod
    def add_ai_message(self, content: str) -> None:
        pass

    @abstractmethod
    def get_messages(self, session_id: str) -> List[Message]:
        """
        Retrieve all messages for a given session ID.
        """
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """
        Clear messages for a given session.
        """
        pass


class LangchainHistoryAdapter(LongTermMemory):
    def __init__(self, history: BaseChatMessageHistory, session_id: str = "default"):
        self.history = history
        self.session_id = session_id

    def add_message(self, message: Message) -> None:
        lc_msg: BaseMessage = message.to_langchain_message()
        self.history.add_message(lc_msg)

    def add_user_message(self, content: str) -> None:
        self.add_message(Message(role=Role.HUMAN, content=content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(Message(role=Role.AI, content=content))

    def get_messages(self, session_id: str = None) -> List[Message]:
        lc_msgs = self.history.messages
        return [from_langchain_message(m) for m in lc_msgs]

    def clear(self, session_id: str = None) -> None:
        self.history.clear()
