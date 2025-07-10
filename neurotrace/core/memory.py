from typing import Any, Dict, List, Union

from langchain.llms.base import BaseLLM
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage
from pydantic import ConfigDict

from neurotrace.core.constants import Role
from neurotrace.core.hippocampus.ltm import LongTermMemory
from neurotrace.core.hippocampus.stm import ShortTermMemory
from neurotrace.core.schema import Message, MessageMetadata


class NeurotraceMemory(BaseMemory):
    """A LangChain-compatible memory implementation using a hybrid memory system.

    This class implements a memory system that combines short-term memory (STM) and
    optional long-term memory (LTM) capabilities. It wraps the ShortTermMemory
    component and integrates with LangChain's memory interface.

    Attributes:
        session_id (str): Unique identifier for the current chat session.
        _stm (ShortTermMemory): Short-term memory component with token limit.
        _ltm (LongTermMemory): Optional long-term memory adapter for persistence.

    Args:
        max_tokens (int, optional): Maximum number of tokens to store in short-term memory.
            Defaults to 2048.
        history (BaseChatMessageHistory, optional): LangChain chat history for long-term
            storage. If provided, enables long-term memory. Defaults to None.
        session_id (str, optional): Identifier for the chat session. Defaults to "default".
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    def __init__(
        self,
        llm: Union[BaseLLM, BaseChatModel],
        session_id: str = "default",
        max_tokens: int = 2048,
        history: BaseChatMessageHistory = None,
    ):
        super().__init__()
        self.llm = llm
        self.session_id = session_id
        self._stm = ShortTermMemory(max_tokens=max_tokens)
        self._ltm = LongTermMemory(history, session_id=session_id) if history else None

    @property
    def memory_variables(self) -> List[str]:
        """Gets the list of memory variables used by this memory component.

        Returns:
            List[str]: List containing "chat_history" as the only memory variable.
        """
        return ["chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
        """Retrieves the current memory state as LangChain messages.

        Converts all messages in short-term memory to LangChain's message format
        for compatibility with the LangChain framework.

        Args:
            inputs (Dict[str, Any]): Input variables (unused in this implementation).

        Returns:
            Dict[str, List[BaseMessage]]: Dictionary with "chat_history" key containing
                the list of messages in LangChain format.
        """
        return {"chat_history": [msg.to_langchain_message() for msg in self._stm.get_messages()]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Saves the conversation context to both short-term and long-term memory.

        Creates Message objects from the input and output and stores them in
        short-term memory. If long-term memory is enabled, also saves to it.

        Args:
            inputs (Dict[str, Any]): Dictionary containing user input with key "input".
            outputs (Dict[str, Any]): Dictionary containing AI output with key "output".
        """
        user_input = inputs.get("input") or ""
        ai_output = outputs.get("output") or ""

        # Build Message objects
        user_msg = Message(
            role=str(Role.HUMAN),
            content=user_input,
            metadata=MessageMetadata(
                session_id=self.session_id,
            ),
        )
        ai_msg = Message(
            role=str(Role.AI),
            content=ai_output,
            metadata=MessageMetadata(
                session_id=self.session_id,
            ),
        )

        # Save in short-term memory
        self._stm.append(user_msg)
        self._stm.append(ai_msg)

        if self._ltm:
            self._ltm.add_message(user_msg)
            self._ltm.add_message(ai_msg)

    def clear(self, delete_history: bool = False) -> None:
        """Clears the memory state.

        Clears the short-term memory and optionally the long-term memory if specified.

        Args:
            delete_history (bool, optional): If True, also clears long-term memory
                if it exists. Defaults to False.
        """
        self._stm.clear()
        if self._ltm and delete_history:
            self._ltm.clear()
