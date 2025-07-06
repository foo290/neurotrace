# neurotrace/adapters/langchain_history_adapter.py
from pydantic import ConfigDict

from langchain_core.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neurotrace.core.hippocampus.stm import ShortTermMemory
from typing import List, Dict, Any

from neurotrace.core.hippocampus.ltm import LangchainHistoryAdapter
from neurotrace.core.schema import Message, MessageMetadata
from neurotrace.core.constants import Role



class NeurotraceMemory(BaseMemory):
    """
    LangChain-compatible memory wrapper that uses ShortTermMemory internally.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    def __init__(self, max_tokens: int = 2048, history: BaseChatMessageHistory = None, session_id: str = "default"):
        super().__init__()
        self.session_id = session_id
        self._stm = ShortTermMemory(max_tokens=max_tokens)
        self._ltm = LangchainHistoryAdapter(history, session_id=session_id) if history else None

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
        """
        Returns the memory in LangChain's message format.
        """
        return {"chat_history": [msg.to_langchain_message() for msg in self._stm.get_messages()]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Receives inputs and outputs from LangChain agent and appends them as messages.
        """
        user_input = inputs.get("input") or ""
        ai_output = outputs.get("output") or ""

        # Build Message objects
        user_msg = Message(
            id='boom',
            role=Role.HUMAN,
            content=user_input,
            metadata=MessageMetadata(
                session_id=self.session_id,
            )
        )
        ai_msg = Message(
            id='boom',
            role=Role.AI,
            content=ai_output,
            metadata=MessageMetadata(
                session_id=self.session_id,
            )
        )

        # Save in short-term memory
        self._stm.append(user_msg)
        self._stm.append(ai_msg)

        if self._ltm:
            self._ltm.add_message(user_msg)
            self._ltm.add_message(ai_msg)

    def clear(self, delete_history: bool = False) -> None:
        self._stm.clear()
        if self._ltm and delete_history:
            self._ltm.clear()
