from typing import List
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest
from newhelm.load_plugins import load_plugins
from newhelm.sut import PromptResponseSUT
from newhelm.general import get_concrete_subclasses


if __name__ == "__main__":
    load_plugins()

    all_suts: List[PromptResponseSUT] = [
        cls() for cls in get_concrete_subclasses(PromptResponseSUT)  # type: ignore[type-abstract]
    ]

    all_tests: List[BasePromptResponseTest] = [
        cls() for cls in get_concrete_subclasses(BasePromptResponseTest)  # type: ignore[type-abstract]
    ]
    for test in all_tests:
        print("\n\nStarting a new test:", test.__class__.__name__)
        # Only have to make the prompt templates once, reusable across SUTs.
        prompt_templates = test.make_prompt_templates()
        for sut in all_suts:
            print("Running sut:", sut.__class__.__name__)
            interactions = []
            for template in prompt_templates:
                # Splitting specialize from evaluate allows us to track the prompts created.
                prompt = sut.specialize(template)
                interaction = sut.evaluate(prompt)
                print("Completed interaction:", interaction)
                interactions.append(interaction)
            # Here is where an annotator would go
            annotated = [
                AnnotatedInteraction(interaction) for interaction in interactions
            ]
            print("Results:", test.calculate_results(annotated))
