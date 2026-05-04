import os
from dotenv import load_dotenv
from langfuse import get_client
from langchain_openai import ChatOpenAI

load_dotenv()

# Step 3: Fetch the prompt from Langfuse
langfuse = get_client()
prompt = langfuse.get_prompt("learning_prompt", label="latest")

# Step 4: Compile with runtime variables
compiled = prompt.compile(language="French", level="A2 beginner")

# Step 5: Build the model from the prompt's config and send messages
model = ChatOpenAI(
    model=prompt.config["model"],
    temperature=prompt.config["temperature"],
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

user_input = "Bonjour! Je veux pratiquer aujourd'hui."
messages = compiled + [{"role": "user", "content": user_input}]

response = model.invoke(messages)

print("\n--- Tutor response ---")
print(response.content)