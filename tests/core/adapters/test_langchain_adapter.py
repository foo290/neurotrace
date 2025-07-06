from neurotrace.core.schema import Message
from neurotrace.core.adapters.langchain_adapter import to_langchain_messages, from_langchain_message
from langchain_core.messages import HumanMessage, AIMessage


def test_to_langchain_messages_conversion():
    msgs = [
        Message(role="user", content="Hi"),
        Message(role="ai", content="Hello!")
    ]
    converted = to_langchain_messages(msgs)

    assert isinstance(converted[0], HumanMessage)
    assert isinstance(converted[1], AIMessage)
    assert converted[0].content == "Hi"
    assert converted[1].content == "Hello!"


def test_from_langchain_message():
    lc_msg = HumanMessage(content="Hey there!")
    msg = from_langchain_message(lc_msg)

    assert msg.role == "user"
    assert msg.content == "Hey there!"
