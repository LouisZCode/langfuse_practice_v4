from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent

load_dotenv()

model = ChatOpenRouter(
    model="nvidia/nemotron-3-super-120b-a12b:free"
)

agent = create_agent(
    model=model
)

response = agent.invoke({"messages" : {"role" : "user", "content" : "In one sentence, what is LLM observability?"}})

print(response)