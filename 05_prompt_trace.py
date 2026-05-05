import os
from langchain_openai import ChatOpenAI
from langfuse import observe, get_client
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

prompt = langfuse.get_prompt("learning_prompt", label="latest")
system_prompt = prompt.compile(language="French", level="A2 beginner")

model = ChatOpenAI(
    model=prompt.config["model"],
    temperature=prompt.config["temperature"],
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    )

@observe(name="tutor_turn")
def run_tutor(user_input: str):
    messages = system_prompt + [{"role": "user", "content": user_input}]

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="tutor-llm-call",
        model=prompt.config["model"],
        input=messages,
        prompt=prompt,
    ) as gen:

        response = model.invoke(messages)
        gen.update(output=response.content)

    return response.content

answer = run_tutor("Bonjour! Je veux pratiquer aujourd'hui.")
print(answer)

langfuse.flush()
