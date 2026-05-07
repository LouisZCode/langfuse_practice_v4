import os
from dotenv import load_dotenv
from langfuse import get_client
from langchain_openai import ChatOpenAI

load_dotenv()
langfuse = get_client()

prompt = langfuse.get_prompt("learning_prompt", label="latest")
compiled = prompt.compile(language="French", level="A2 beginner")

model = ChatOpenAI(
    model=prompt.config["model"],
    temperature=prompt.config["temperature"],
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

messages = compiled + [{"role": "user", "content": "Bonjour, comment ça va?"}]

with langfuse.start_as_current_observation(
    as_type="generation",
    name="tutor-llm-call",
    model=prompt.config["model"],
    input=messages,
    prompt=prompt,
) as gen:
    response = model.invoke(messages)
    gen.update(
        output=response.content,
        usage_details={
            "input": response.usage_metadata["input_tokens"],
            "output": response.usage_metadata["output_tokens"],
            "total": response.usage_metadata["total_tokens"],
        },
    )

print(response.content)
langfuse.flush()