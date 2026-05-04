from langfuse import observe, get_client
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import os

load_dotenv()

#Fetch the Prompt:

langfuse = get_client()

prompt = langfuse.get_prompt("learning_prompt", label="latest")

print(prompt)

print(prompt.prompt)        # the text body (or message list for Chat type)
print(prompt.config)        # the config dict you set
print(prompt.version)       # version number
print(prompt.labels)        # list of labels
print(prompt.name)          

exit()

langfuse_handler = CallbackHandler()

model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    )

agent = create_agent(model=model)

response = agent.invoke({"messages": [{"role": "user", "content": "capital of France"}]})

print()
print(response["messages"][-1].content)
print()