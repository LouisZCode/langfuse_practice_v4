# Here we simply added an observe decorator to the functions we want to follow in case they are also important
# to "log" inside langfuse

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
agent = create_agent(model=model)


@observe(name="function_observation")
def transform_question(raw: str) -> str:
    return raw.strip().capitalize()


user_input = input("Ask your question:\n")
transformed = transform_question(user_input)

response = agent.invoke(
    {"messages": [{"role": "user", "content": transformed}]},
    config={
    "callbacks": [langfuse_handler],
    "run_name" : "agent_trace_renamed",
    "metadata" : {
        "langfuse_user_id": "luis_testing_user",
        "langfuse_session_id": "001",
        "langfuse_tags": ["module-3-final", "observe"],
    }
    }, 
)

print(response["messages"][-1].content)

langfuse.flush()  # ← so the trace actually arrives before the script exits