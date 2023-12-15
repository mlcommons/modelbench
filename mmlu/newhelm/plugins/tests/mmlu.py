from typing import List
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest, BaseTest
from newhelm.placeholders import PromptTemplate, Result


class MMLU(BasePromptResponseTest):
    def make_prompt_templates(self) -> List[PromptTemplate]:
        # In the real thing, this would use an ExampleImporter and Adapters
        return [
            PromptTemplate(
                eval_instance_block="When I think of MMLU, the word that comes to mind is"
            ),
            PromptTemplate(
                eval_instance_block="But the worst part is when",
            ),
        ]

    def calculate_results(
        self, interactions: List[AnnotatedInteraction]
    ) -> List[Result]:
        # In the real thing, this would be handled by Metric objects
        return [Result("count", value=len(interactions))]
