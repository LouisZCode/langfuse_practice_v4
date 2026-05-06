import os
from dotenv import load_dotenv
from langfuse import get_client, Evaluation
from langchain_openai import ChatOpenAI

load_dotenv()
langfuse = get_client()

prompt = langfuse.get_prompt("learning_prompt", label="latest")

model = ChatOpenAI(
    model=prompt.config["model"],
    temperature=prompt.config["temperature"],
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

dataset = langfuse.get_dataset("tutor-eval")
run_name = f"v{prompt.version}-with-LLM-as-Judge"


def my_task(*, item, **kwargs):
    compiled = prompt.compile(language="French", level="A2 beginner")
    messages = compiled + [{"role": "user", "content": item.input}]

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
    return response.content


def has_french_chars(*, input, output, expected_output, metadata, **kwargs):
    french_chars = set("éèêëàâäîïôöùûüçœæ")
    found = any(c in french_chars for c in output.lower())
    return Evaluation(
        name="has_french_chars",
        value=found,
        comment=f"French characters {'found' if found else 'NOT found'} in output",
    )


def has_translation(*, input, output, expected_output, metadata, **kwargs):
    found = "(" in output and ")" in output
    return Evaluation(
        name="has_translation",
        value=found,
        comment=f"Parentheses {'found' if found else 'NOT found'} (proxy for English translation)",
    )


dataset.run_experiment(
    name=run_name,
    description=f"Tutor prompt v{prompt.version} with heuristic evals",
    task=my_task,
    evaluators=[has_french_chars, has_translation],
)

langfuse.flush()
print(f"Run '{run_name}' complete.")