import string
from typing import Optional, Sequence

from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.cost import RealizedCost
from modelgauge.annotators.composer.nodes import Enricher
from modelgauge.config import load_secrets_from_config
from modelgauge.model_options import ModelOptions
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import PromptResponseSUT, TextPrompt
from modelgauge.sut_factory import SUT_FACTORY


class PromptEngineeredNode(Enricher):
    """Node that enriches the context by making an LLM call with a prompt template."""

    def __init__(
        self,
        name: str,
        routes: Sequence[str],
        prompt_template: string.Template,
        sut_id: str,
        model_options=None,
        sut_secrets: Optional[RawSecrets] = None,
        **sut_kwargs,
    ) -> None:
        super().__init__(name=name, routes=routes)

        subs = prompt_template.get_identifiers()
        if not set(subs).issubset({"prompt", "response"}):
            raise ValueError("Prompt template may only have 'prompt' and 'response' placeholders.")
        self.prompt_template = prompt_template

        if model_options is None:
            model_options = ModelOptions()
        self.model_options = model_options

        if sut_secrets is None:
            sut_secrets = load_secrets_from_config()
        sut = SUT_FACTORY.make_instance(uid=sut_id, secrets=sut_secrets, **sut_kwargs)
        if not isinstance(sut, PromptResponseSUT):
            raise ValueError(
                f"PromptEngineeredAnnotator only works with PromptResponseSUTs. SUT {sut_id} is of type {type(sut)}"
            )
        self.sut: PromptResponseSUT = sut

    def _build_prompt(self, ctx: EvalContext) -> TextPrompt:
        return TextPrompt(text=self.prompt_template.safe_substitute(prompt=ctx.prompt, response=ctx.response))

    def _count_tokens(self, text: str) -> int:
        # Simple tokenizer.
        return len(text.split())

    def run(self, ctx: EvalContext) -> NodeOutput:
        prompt = self._build_prompt(ctx)
        sut_request = self.sut.translate_text_prompt(prompt, options=self.model_options)
        resp = self.sut.evaluate(sut_request)
        sut_response = self.sut.translate_response(sut_request, resp)
        return NodeOutput(
            value=sut_response.text,
            realized_cost=RealizedCost(
                input_token_cost=self._count_tokens(prompt.text),
                output_token_cost=self._count_tokens(sut_response.text),
            ),
            original_ctx=ctx,
            updated_ctx=ctx.with_response(sut_response.text),
        )
