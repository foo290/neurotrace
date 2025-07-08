from neurotrace.core.schema import Message
from neurotrace.neurotrace_logging.logger_factory import get_logger

logger = get_logger("neurotrace.memory")


class MemoryLogger:
    @staticmethod
    def log_add(message: Message, destination: str):
        logger.info(
            f"Added to {destination.upper()}: "
            f"{message.role} - {message.content[:60]!r} "
            f"(tokens ~{message.estimated_token_length()}, id={message.id})"
        )

    @staticmethod
    def log_evict(message: Message):
        logger.warning(f"Evicted from STM: {message.role} - {message.content[:60]!r} " f"(id={message.id})")

    @staticmethod
    def log_search(query: str, results: list[Message]):
        logger.info(f"Vector search for: {query!r} — {len(results)} results returned.")

    @staticmethod
    def log_clear(target: str):
        logger.info(f"Cleared messages from {target.upper()}")

    @staticmethod
    def log_error(message: str):
        logger.error(message)
