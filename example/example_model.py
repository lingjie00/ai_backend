"""Example usage of the AI backend package with a simple model configuration."""

# %%
from pydantic import BaseModel

from ai_backend import LangChainClient, PromptLoader


class StoryBoard(BaseModel):
    """Data model representing a storyboard for a story generation task."""

    title: str
    setting: str
    characters_name: str


loader = PromptLoader(".")
client = LangChainClient(
    prompt_loader=loader,
    model_name="example_model",
    structured_output_model=StoryBoard,
    # we show how different keywords can be used to specify the role of the
    # message in the prompt
    additional_prompts=[("user", "{context}")],
)
model = client.model

# %%
context = {"context": "Write a short story about a brave knight in a magical kingdom."}
output = model.invoke(input=context)
output_class = StoryBoard.model_validate(output)
# return pydantic model instance
print(output_class.model_dump_json(indent=2))

# %%
chat_output = client.chat_model.invoke(context)
# return message
chat_output.pretty_print()
