# neurotrace/core/_stm.py
import uuid

from abc import ABC, abstractmethod
from typing import List
from neurotrace.core.schema import Message


class BaseShortTermMemory(ABC):
    @abstractmethod
    def append(self, message: Message) -> None:
        ...

    @abstractmethod
    def get_messages(self) -> List[Message]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def set_messages(self, messages: List[Message]) -> None:
        ...

    @abstractmethod
    def total_tokens(self) -> int:
        ...

    def __len__(self) -> int:
        return len(self.get_messages())



class ShortTermMemory(BaseShortTermMemory):
    def __init__(self, max_tokens: int = 2048):
        self.messages: List[Message] = []
        self.max_tokens = max_tokens

    def append(self, message: Message) -> None:
        if not message.id:
            message.id = str(uuid.uuid4())

        self.messages.append(message)
        self._evict_if_needed()

    def get_messages(self) -> List[Message]:
        return self.messages

    def clear(self) -> None:
        self.messages = []

    def _evict_if_needed(self) -> None:
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
        self.messages = messages
        self._evict_if_needed()

    def total_tokens(self) -> int:
        return sum(m.estimated_token_length() for m in self.messages)

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return f"<STM messages={len(self.messages)} tokens={self.total_tokens()}/{self.max_tokens}>"
