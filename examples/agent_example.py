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
