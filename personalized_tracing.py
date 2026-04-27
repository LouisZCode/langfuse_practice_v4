import os
import uuid
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

# Simulate one user having a conversation of 3 turns
user_id = "luis-test-01"
session_id = str(uuid.uuid4())   # one UUID for the whole "conversation"

questions = [
    "What's the capital of France?",
    "what is observability in LLMs?",
    "what is Langfuse?",
]

for i, q in enumerate(questions):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={
            "callbacks": [langfuse_handler],
            "run_name": f"chat-turn-{i+1}",   # ← custom trace name
            "metadata": {
                "langfuse_user_id": user_id,
                "langfuse_session_id": session_id,
                "langfuse_tags": ["module-3-test", "general-qa"],
                "question_index": str(i + 1),
            },
        },
    )
    print(f"\n--- Turn {i+1}: {q}\n{response['messages'][-1].content[:150]}...")