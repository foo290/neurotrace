from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from neurotrace.core.memory import NeurotraceMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Tool that conforms to LangChain's expectations
@tool
def get_mood_tip(dummy: str) -> str:
    """Give a random wellness tip. Ignores input."""
    return "Take a walk and listen to calming music."

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Use NeurotraceMemory for short-term context
memory = NeurotraceMemory(max_tokens=100)

# Agent setup
agent = initialize_agent(
    tools=[get_mood_tip],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Driver loop
if __name__ == "__main__":
    print("Neurotrace Agent (Gemini). Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        response = agent.invoke({"input": user_input})
        print("Agent:", response["output"])
        print("-- Memory State --")
        print("Total Messages:", len(memory._stm.get_messages()))
        print("total tokens:", memory._stm.total_tokens())
        print("Current Memory:", memory._stm.get_messages())
        print("-------------------")


