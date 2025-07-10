# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import tool
# from neurotrace.core.memory import NeurotraceMemory
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# # Tool that conforms to LangChain's expectations
# @tool
# def get_mood_tip(dummy: str) -> str:
#     """Give a random wellness tip. Ignores input."""
#     return "Take a walk and listen to calming music."
#
# # Initialize LLM
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
#
# # Use NeurotraceMemory for short-term context
# memory = NeurotraceMemory(max_tokens=100)
#
# # Agent setup
# agent = initialize_agent(
#     tools=[get_mood_tip],
#     llm=llm,
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     memory=memory,
#     verbose=True
# )
#
# # Driver loop
# if __name__ == "__main__":
#     print("Neurotrace Agent (Gemini). Type 'exit' to quit.")
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.strip().lower() == "exit":
#             break
#         response = agent.invoke({"input": user_input})
#         print("Agent:", response["output"])
#         print("-- Memory State --")
#         print("Total Messages:", len(memory._stm.get_messages()))
#         print("total tokens:", memory._stm.total_tokens())
#         print("Current Memory:", memory._stm.get_messages())
#         print("-------------------")
#
#

import os

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from neurotrace.core.memory import NeurotraceMemory
from neurotrace.core.schema import Message
from neurotrace.core.tools.memory import save_memory_tool
from neurotrace.core.tools.system import get_system_tools_list
from neurotrace.core.tools.vector import vector_memory_search_tool
from neurotrace.core.utils import load_prompt  # Assuming prompt loader
from neurotrace.core.vector_memory import VectorMemoryAdapter  # Your implementation

# Load environment variables
load_dotenv()
print(os.environ.get("TAVILY_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Setup memory
memory = NeurotraceMemory(max_tokens=100, llm=llm)

# Setup vector store (Chroma in this example)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(embedding_function=embedding_model, persist_directory=".chromadb")

vector_memory = VectorMemoryAdapter(vectorstore)

mem_search_tool = vector_memory_search_tool(
    vector_memory_adapter=vector_memory,
)
# mem_save_tool = vector_memory_save_tool(vector_memory_adapter=vector_memory)

from langchain_community.graphs.neo4j_graph import Neo4jGraph

from neurotrace.core.hippocampus.memory_orchestrator import MemoryOrchestrator

graph_store = Neo4jGraph(
    url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
    username=os.environ.get("NEO4J_USERNAME", "neo4j"),
    password=os.environ.get("NEO4J_PASSWORD", "test1234"),
)


mem_orchestrator = MemoryOrchestrator(
    llm=llm,
    vector_store=vectorstore,
    graph_store=graph_store,
)
mem_save_tool = save_memory_tool(
    memory_orchestrator=mem_orchestrator,
)

# Agent setup
agent = initialize_agent(
    tools=[mem_search_tool, mem_save_tool, *get_system_tools_list()],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

# Driver loop
if __name__ == "__main__":
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
        # vector_memory.add_messages([user_msg, ai_msg])

        # Debug Memory
        print("-- Memory State --")
        print("STM Messages:", len(memory._stm.get_messages()))
        print("STM Tokens:", memory._stm.total_tokens())
        print("------------------")
