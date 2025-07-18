# Neurotrace


[![PyPI version](https://img.shields.io/pypi/v/neurotrace)](https://pypi.org/project/neurotrace/)

A hybrid memory library designed for LangChain agents, providing dual-layer memory architecture with short-term buffer memory and long-term hybrid RAG system capabilities.

## Overview

Neurotrace provides persistent, intelligent memory for conversational agents that improves over time and enables contextual understanding and recall. It combines vector-based and graph-based RAG (Retrieval Augmented Generation) systems to provide deeper and more accurate contextual reasoning.

## 🎯 Key Features

- **Dual-Layer Memory Architecture**
  - Short-term buffer memory for immediate context
  - Long-term hybrid RAG system for persistent storage

- **Real-time Processing**
  - Real-time recall during conversations
  - Intelligent storage and compression

- **Rich Message Structure**
  - Custom metadata-rich message formats
  - Support for filtering and semantic tracing

- **Hybrid Retrieval System**
  - Combined vector and graph-based RAG
  - Enhanced contextual reasoning capabilities

## Graph db integration (Graph RAG)

![neo4j.png](https://github.com/foo290/neurotrace/blob/main/readme_assets/images/neo4j.png)

## 🎯 Target Users

- Developers building AI agents with LangChain
- Researchers exploring memory augmentation in LLMs
- Enterprises deploying context-aware AI assistants

## Quick Start

### Installation

```bash
pip install neurotrace
```

### Complete Example

A complete, runnable example is available in `examples/agent_example.py`. This example demonstrates:
- Setting up a Neurotrace agent with both short-term and long-term memory
- Configuring vector and graph storage
- Implementing an interactive conversation loop
- Monitoring memory usage

To run the example:
```bash
# First set up your environment variables
export NEO4J_URL=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
export GOOGLE_API_KEY=your_google_api_key

# Then run the example
python examples/agent_example.py
```

### Required Environment Variables

```bash
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GOOGLE_API_KEY=your_google_api_key  # For Gemini LLM
```

## Technical Documentation

### Core Schema

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

### Usage

```python
"""
A complete example of implementing a Neurotrace-powered agent with both short-term and long-term memory.
"""

import os

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from neurotrace.core.hippocampus.memory_orchestrator import MemoryOrchestrator
from neurotrace.core.memory import NeurotraceMemory
from neurotrace.core.schema import Message
from neurotrace.core.tools.memory import memory_search_tool, save_memory_tool
from neurotrace.core.tools.system import get_system_tools_list


def setup_agent():
    """Initialize and configure the Neurotrace agent with memory components."""

    # Load environment variables
    load_dotenv()

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Setup short-term memory
    memory = NeurotraceMemory(max_tokens=100, llm=llm)

    # Setup vector store
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(embedding_function=embedding_model, persist_directory=".chromadb")

    # Setup graph database
    graph_store = Neo4jGraph(
        url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
    )

    # Initialize Memory Orchestrator
    mem_orchestrator = MemoryOrchestrator(
        llm=llm,
        vector_store=vectorstore,
        graph_store=graph_store,
    )

    # Setup memory tools
    mem_save_tool = save_memory_tool(memory_orchestrator=mem_orchestrator)
    mem_search_tool = memory_search_tool(memory_orchestrator=mem_orchestrator)

    # Initialize Agent
    agent = initialize_agent(
        tools=[mem_search_tool, mem_save_tool, *get_system_tools_list()],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

    return agent, memory


def run_agent():
    """Run the agent in an interactive conversation loop."""

    agent, memory = setup_agent()

    print("Neurotrace Agent Ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break

        # Process user input
        response = agent.invoke({"input": user_input})
        output = response["output"]
        print("Agent:", output)

        # Save conversation to memory
        user_msg = Message(role="human", content=user_input)
        ai_msg = Message(role="ai", content=output)

        # Debug Memory State
        print("\n-- Memory State --")
        print("STM Messages:", len(memory._stm.get_messages()))
        print("STM Tokens:", memory._stm.total_tokens())
        print("------------------\n")


if __name__ == "__main__":
    run_agent()

```
