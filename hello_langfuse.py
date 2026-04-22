import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

agent = create_agent(model=model)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "In one sentence, what is LLM observability?"}]},
    config={"callbacks": [langfuse_handler]},
)

print(response)