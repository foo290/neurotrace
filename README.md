# neurotrace

> Work in progress

Neurotrace is a Python library designed to facilitate the development of AI applications with a focus on memory management, message handling, and integration with various data storage systems.
It provides a structured approach to managing conversational data, enabling developers to build intelligent systems that can remember context, emotions, and user interactions.


## Core Schema

The `neurotrace.core.schema` module defines the fundamental data structures used throughout the project.

### Message

The core Message class represents a single message in the system:

```python
from neurotrace.core.schema import Message, MessageMetadata, EmotionTag

message = Message(
    role="user",           # Can be "user", "ai", or "system"
    content="Hello!",      # The message text content
    metadata=MessageMetadata(
        source="chat",
        emotions=EmotionTag(sentiment="positive")
    )
)
```

Key features of Message:
- Auto-generated UUID for each message
- Automatic timestamp on creation
- Type-safe role validation
- Rich metadata support via MessageMetadata

### Message Components

#### EmotionTag

Represents the emotional context of a message:

```python
from neurotrace.core.schema import EmotionTag

emotion = EmotionTag(
    sentiment="positive",  # Can be "positive", "neutral", or "negative"
    intensity=0.8         # Optional float value indicating intensity
)
```

#### MessageMetadata

Contains additional information and context about a message:

```python
from neurotrace.core.schema import MessageMetadata, EmotionTag

metadata = MessageMetadata(
    token_count=150,                    # Number of tokens in the message
    embedding=[0.1, 0.2, 0.3],         # Vector embedding for similarity search
    source="chat",                      # Source: "chat", "web", "api", or "system"
    tags=["important", "follow-up"],    # Custom tags
    thread_id="thread_123",            # Conversation thread identifier
    user_id="user_456",               # Associated user identifier
    related_ids=["msg_789"],          # Related message IDs
    emotions=EmotionTag(sentiment="positive"),  # Emotional context
    compressed=False                   # Compression status
)
```

Each field in MessageMetadata is optional and provides specific context:
- `token_count`: Used for tracking token usage
- `embedding`: Vector representation for similarity search
- `source`: Indicates message origin
- `tags`: Custom categorization
- `thread_id`: Groups messages in conversations
- `user_id`: Links messages to users
- `related_ids`: Connects related messages
- `emotions`: Captures emotional context
- `compressed`: Indicates if content is compressed

## Adapters Module

The `neurotrace.core.adapters` module provides utilities for converting Message objects to and from various database and framework formats.

### Vector Database Adapter

Convert messages to a format suitable for vector database storage:

```python
from neurotrace.core.schema import Message, MessageMetadata
from neurotrace.core.adapters.vector_db_adapter import to_vector_record

# Create a message with an embedding
message = Message(
    content="Hello world",
    metadata=MessageMetadata(
        embedding=[0.1, 0.2, 0.3],
        tags=["greeting"]
    )
)

# Convert to vector DB format
record = to_vector_record(message)
# Results in: {
#     "id": "<uuid>",
#     "text": "Hello world",
#     "embedding": [0.1, 0.2, 0.3],
#     "metadata": {"tags": ["greeting"], ...}
# }
```

### Graph Database Adapter

Convert messages to graph nodes and relationships:

```python
from neurotrace.core.adapters.graph_db_adapter import to_graph_node, graph_edges_from_related_ids
from neurotrace.core.schema import Message, MessageMetadata

# Create a message with related IDs
message = Message(
    content="Follow-up response",
    metadata=MessageMetadata(
        related_ids=["msg123"],
        tags=["follow-up"]
    )
)

# Convert to graph node
node = to_graph_node(message)
# Results in: {
#     "id": "<uuid>",
#     "labels": ["Message"],
#     "properties": {
#         "role": "user",
#         "content": "Follow-up response",
#         "tags": ["follow-up"],
#         ...
#     }
# }

# Generate relationship edges
edges = graph_edges_from_related_ids(message)
# Results in: [{
#     "from": "<current_msg_id>",
#     "to": "msg123",
#     "type": "RELATED_TO"
# }]
```

### LangChain Adapter

Convert between neurotrace Messages and LangChain message types:

```python
from neurotrace.core.adapters.langchain_adapter import from_langchain_message
from langchain_core.messages import HumanMessage

# Convert from LangChain to neurotrace format
lc_msg = HumanMessage(content="Hey there!")
msg = from_langchain_message(lc_msg)
# Results in Message(role="user", content="Hey there!")
```

Each adapter provides type-safe conversion between neurotrace's Message format and the target system's format, ensuring compatibility with various databases and frameworks.

## Memory Management

The `neurotrace` system implements a sophisticated memory management system inspired by human memory architecture.

### Short Term Memory (STM)

Located in `neurotrace.core.hippocampus.stm`, the `ShortTermMemory` class provides temporary message storage with the following features:

```python
from neurotrace.core.hippocampus.stm import ShortTermMemory

# Initialize with token limit
stm = ShortTermMemory(max_tokens=50)

# Add messages
stm.append(message)  # automatically handles token budget

# Retrieve all current messages
messages = stm.get_messages()

# Clear memory
stm.clear()
```

Key features:
- Token-based memory management
- Automatic message eviction when token limit is exceeded
- Timestamp-based message tracking
- Efficient message retrieval

### LangChain Integration

The `NeurotraceMemory` class provides seamless integration with LangChain:

```python
from neurotrace.core.memory import NeurotraceMemory

memory = NeurotraceMemory(max_tokens=20)

# Works with LangChain's memory interface
memory.save_context({"input": "What's your name?"}, {"output": "I'm Neurotrace."})
history = memory.load_memory_variables({})

# Access chat history
messages = history["chat_history"]  # Returns LangChain message format
```

Features:
- Compatible with LangChain's memory system
- Automatic conversion between Neurotrace and LangChain message formats
- Preserves all metadata during conversions
- Token budget management
- Supports message source tracking

### Vector Database Integration

The vector database adapter now supports:
- Automatic embedding storage and retrieval
- Metadata preservation in vector records
- Custom tagging for efficient retrieval
- UUID-based message tracking

Example usage:
```python
from neurotrace.core.adapters.vector_db_adapter import to_vector_record

record = to_vector_record(message)
# Creates a vector record with:
# - Unique ID
# - Message content
# - Embeddings
# - Complete metadata (tags, source, timestamps, etc.)
```

## Usage with LangChain

Neurotrace can be seamlessly integrated with LangChain for building conversational agents with both short-term and vector memory capabilities:

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import Chroma
from neurotrace.core.memory import NeurotraceMemory
from neurotrace.core.vector_memory import VectorMemoryAdapter
from neurotrace.core.tools.vector import vector_memory_search_tool
from neurotrace.core.schema import Message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Setup short-term memory
memory = NeurotraceMemory(max_tokens=100)

# Setup vector store with Chroma
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(embedding_function=embedding_model, persist_directory=".chromadb")
vector_memory = VectorMemoryAdapter(vectorstore, embedding_model)

# Create memory search tool
mem_search_tool = vector_memory_search_tool(
    vector_memory_adapter=vector_memory,
)

# Agent setup
agent = initialize_agent(
    tools=[mem_search_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
```

The above example demonstrates:
- Integration with LangChain's agent system
- Short-term memory management with NeurotraceMemory
- Vector store setup using Chroma for long-term memory
- Custom memory search tool integration
- LLM configuration with Google's Gemini model

You can run the agent in an interactive loop with vector memory storage:

```python
print("Neurotrace Agent (Gemini + Vector Memory). Type 'exit' to quit.")
while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() == "exit":
        break

    response = agent.invoke({"input": user_input})
    output = response["output"]
    print("Agent:", output)

    # Save both user and AI messages into vector memory
    user_msg = Message(role="human", content=user_input)
    ai_msg = Message(role="ai", content=output)
    vector_memory.add_messages([user_msg, ai_msg])
```

The system automatically manages:
- Short-term memory token limits
- Vector embeddings for long-term storage
- Conversation history in both memory systems
- Memory search capabilities through custom tools
