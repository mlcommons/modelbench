import json
from typing import Optional, Union

from pydantic import BaseModel

from modelgauge.annotator import CompletionAnnotator
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTOptions, SUTResponse
from modelgauge.sut_factory import SUT_FACTORY
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIChat, OpenAIOrgId
from modelgauge.suts.together_client import TogetherApiKey, TogetherChatSUT

# FIXME - To use the openai GPT-4 annotator, add the "ChatRole.system: _SYSTEM_ROLE", role into the _ROLE_MAP in modelgauge/suts/openai_client.py:
# LINE 32:
#
# _ROLE_MAP = {
#     ChatRole.user: _USER_ROLE,
#     ChatRole.sut: _ASSISTANT_ROLE,
#     ChatRole.system: _SYSTEM_ROLE,   ##### This line needs to be added into _ROLE_MAP
# }


class PromptEngineeredAnnotatorConfig(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 4000
    name: str
    description: str
    llm_type: str
    llm_path: str
    labeler_system_prompt: str
    labeler_user_prompt: str
    taxonomy: str
    role_to_measure: str


class PromptEngineeredAnnotator(CompletionAnnotator[SafetyModelResponse]):
    """Annotator using prompt engineered Together model for hazard and safety classification"""

    def __init__(
        self,
        uid: str,
        config: PromptEngineeredAnnotatorConfig,
        api_key: Union[TogetherApiKey, OpenAIApiKey],
        org_id: Optional[OpenAIOrgId] = None,
    ):
        super().__init__(uid)
        self.config = config
        self.api_key = api_key
        # TODO: Is this the best way to handle different base model types?
        self.model: Union[TogetherChatSUT, OpenAIChat]
        if self.config.llm_type == "together":
            assert isinstance(
                self.api_key, TogetherApiKey
            ), "API key must be of type TogetherApiKey for Together models."
            self.model = TogetherChatSUT(uid, self.config.llm_path, self.api_key)
        elif self.config.llm_type == "openai":
            assert isinstance(self.api_key, OpenAIApiKey), "API key must be of type OpenAIApiKey for OpenAI models."
            assert org_id is not None, "OpenAIOrgId must be provided for OpenAI models."
            self.org_id = org_id
            self.model = OpenAIChat(
                "annotator",
                "gpt-4",  # TODO: llm_path?
                self.api_key,
                self.org_id,
            )
        elif self.config.llm_type == "registry":
            self.model = SUT_FACTORY.make_instance(self.config.llm_path, secrets=self.api_key)  # type: ignore
        else:
            raise ValueError(f"Unsupported LLM type: {self.config.llm_type}")

    @staticmethod
    def _prompt_to_pass(test_item: TestItem) -> str:
        assert isinstance(test_item.prompt, TextPrompt)
        return test_item.prompt.text

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        """Convert the prompt+completion into the native representation for this annotator."""
        system_prompt = ChatMessage(
            text=self.config.labeler_system_prompt.format(taxonomy=self.config.taxonomy),
            role=ChatRole.system,
        )

        # We have a list of interaction messages, simplest option is to turn into 'role': 'message' block of text
        if isinstance(test_item.prompt, ChatPrompt):
            # This isn't used by either benchmark.
            chat_interactions_to_annotate = flatten_conversation_to_str(test_item.prompt)
        elif isinstance(test_item.prompt, TextPrompt):
            assert test_item.prompt.text is not None, "TextPrompt must have text."
            assert response.text is not None, "SUTResponse must have text."
            chat_interactions_to_annotate = f"User: {self._prompt_to_pass(test_item)}\n\nAssistant: {response.text}"
        else:
            raise ValueError(f"Can only handle ChatPrompt, TextPrompt types. Type given: {type(test_item.prompt)}")

        request = self.model.translate_chat_prompt(  # type: ignore
            prompt=ChatPrompt(
                messages=[
                    system_prompt,
                    ChatMessage(
                        text=self.config.labeler_user_prompt.format(
                            messages=chat_interactions_to_annotate,
                            role_to_measure=self.config.role_to_measure,
                        ),
                        role=ChatRole.user,
                    ),
                ]
            ),
            options=SUTOptions(max_tokens=self.config.max_tokens, temperature=self.config.temperature),
        )

        return request

    def annotate(self, annotation_request):
        """Perform annotation and return the raw response from the annotator."""
        result = self.model.evaluate(annotation_request)
        return result

    def translate_response(self, request, response) -> SafetyModelResponse:
        """Convert the raw response into the form read by Tests."""
        translated_response: SUTResponse = self.model.translate_response(request, response)
        response_text = translated_response.text

        # Parse response
        try:
            # Clean up the response and isolate the JSON string
            json_marker = "{"
            start_index = response_text.index(json_marker)
            cleaned_str = response_text[start_index:].replace("\n", "")
            end_index = cleaned_str.rfind("}")
            if end_index != -1:
                cleaned_str = cleaned_str[: end_index + 1]

            # Parse the cleaned string
            response_json = json.loads(cleaned_str)

            result = SafetyModelResponse(**response_json)
            # result.total_tokens_used = response.usage.total_tokens
            result.raw_response = response_text
            return result
        except Exception as e:
            print(f"Error parsing response JSON: {e}")
            return SafetyModelResponse(
                is_safe=False,
                safety_categories=[],
                is_valid=False,
                raw_response=response_text,
            )


class PromptEngineeredSecurityAnnotator(PromptEngineeredAnnotator):
    """Pass the seed prompt to the annotator instead of the actual attack prompt that is passed to the SUT."""

    @staticmethod
    def _prompt_to_pass(test_item: TestItem) -> str:
        try:
            prompt = test_item.context.seed_prompt
        except AttributeError:
            raise ValueError("Can only run security annotator on test items with `seed_prompt` in their context")
        return prompt


def flatten_conversation_to_str(chat: ChatPrompt, *, user_role: str = "User", sut_role: str = "Assistant") -> str:
    """Flattens a chat conversation into a single text prompt"""
    blocks = []
    for message in chat.messages:
        role_text: str
        if message.role == ChatRole.user:
            role_text = user_role
        else:
            role_text = sut_role
        blocks.append(f"{role_text}: {message.text}")
    return "\n\n".join(blocks)
