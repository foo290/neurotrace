from neurotrace.core.hippocampus.stm import ShortTermMemory
from neurotrace.core.schema import Message

stm = ShortTermMemory(max_tokens=100)

msg1 = Message(role="user", content="Hello")
msg2 = Message(role="ai", content="Hi there!")
stm.append(msg1)
stm.append(msg2)

print(stm.get_messages())
print(stm.messages[0].to_human_message())

