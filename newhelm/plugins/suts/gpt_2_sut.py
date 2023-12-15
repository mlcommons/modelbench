from typing import List
from newhelm.placeholders import (
    LocalWindowService,
    Prompt,
    PromptTemplate,
    PlaceholderTokenizer,
    WindowServiceConfig,
)
from newhelm.sut import Interaction, PromptResponseSUT, Turn


def template_to_string(prompt_template: PromptTemplate):
    """The idea is each SUT can have its own definition for this, but we can have libraries for the common way to do it."""
    instructions = [prompt_template.instructions_block]
    if instructions == [""]:
        instructions = []
    blocks: List[str] = (
        instructions
        + prompt_template.train_instance_blocks
        + [prompt_template.eval_instance_block]
    )
    return "\n".join(blocks)


class GPT2(PromptResponseSUT):
    """The SUT should have all the details currently spread across model_deployment and model_metadata."""

    def __init__(self):
        self.window_service = LocalWindowService(
            PlaceholderTokenizer(),
            WindowServiceConfig(
                max_sequence_length=1024,
            ),
        )

    def specialize(self, prompt_template: PromptTemplate) -> Prompt:
        """The SUT is responsible for making the PromptTemplate work."""
        prompt_text = template_to_string(prompt_template)
        if self.window_service.fits_within_context_window(prompt_text):
            return Prompt(prompt_text, truncated=False)
        prompt_text = self.window_service.truncate_training_then_from_right(
            prompt_template, template_to_string
        )
        return Prompt(text=prompt_text, truncated=True)

    def evaluate(self, prompt: Prompt) -> Interaction:
        # Pure placeholder.
        number_of_words = len(prompt.text.split())
        return Interaction([Turn(prompt, f"The prompt has {number_of_words} words.")])
