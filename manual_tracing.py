from langfuse import observe
import os
from dotenv import load_dotenv
import uuid
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from urllib3 import response

load_dotenv()

langfuse_handler = CallbackHandler()

model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    )

agent = create_agent(model=model)


question = input("ask your question: \n")

@observe
def transform_question():
    new_question = "what is the capital of France?"
    return new_question

response = agent.invoke(
    {"messages": [{"role": "user", "content": transform_question()}]}
    )

print(response)