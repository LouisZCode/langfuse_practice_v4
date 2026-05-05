import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

load_dotenv()

#This loads the env langfuse elements and creates the opentelemetry processor
#Handler makes the langfuse call.
langfuse_handler = CallbackHandler()

#Call openrouter using the Openai SDK
model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

#Wrapper, will create extra layers internally
agent = create_agent(model=model)


while True:

    question = input("\nHere your question:\n")

    if question == "exit":
        break

    else:

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"callbacks": [langfuse_handler]},
        )

        print(response)