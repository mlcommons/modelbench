import json
import re
from typing import Optional, Union

from pydantic import BaseModel

from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.log_config import get_logger
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.sut import SUTOptions, SUTResponse
from modelgauge.sut_factory import SUT_FACTORY
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIChat, OpenAIOrganization
from modelgauge.suts.together_client import TogetherApiKey, TogetherChatSUT

# FIXME - To use the openai GPT-4 annotator, add the "ChatRole.system: _SYSTEM_ROLE", role into the _ROLE_MAP in modelgauge/suts/openai_client.py:
# LINE 32:
#
# _ROLE_MAP = {
#     ChatRole.user: _USER_ROLE,
#     ChatRole.sut: _ASSISTANT_ROLE,
#     ChatRole.system: _SYSTEM_ROLE,   ##### This line needs to be added into _ROLE_MAP
# }

logger = get_logger(__name__)
_REGEX_IS_SAFE = re.compile(r'"is_safe"\s*:\s*(true|false)', flags=re.IGNORECASE)


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


class PromptEngineeredAnnotator(Annotator):
    """Annotator using prompt engineered Together model for hazard and safety classification"""

    def __init__(
        self,
        uid: str,
        config: PromptEngineeredAnnotatorConfig,
        api_key: Union[TogetherApiKey, OpenAIApiKey],
        organization: Optional[OpenAIOrganization] = None,
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
            self.organization = organization
            self.model = OpenAIChat(
                "annotator",
                "gpt-4",  # TODO: llm_path?
                self.api_key,
                self.organization,
            )
        elif self.config.llm_type == "registry":
            self.model = SUT_FACTORY.make_instance(self.config.llm_path, secrets=self.api_key)  # type: ignore
        else:
            raise ValueError(f"Unsupported LLM type: {self.config.llm_type}")

    def translate_prompt(self, prompt: TextPrompt | ChatPrompt, response: SUTResponse):
        """Convert the prompt+completion into the native representation for this annotator."""
        system_prompt = ChatMessage(
            text=self.config.labeler_system_prompt.format(taxonomy=self.config.taxonomy),
            role=ChatRole.system,
        )

        # We have a list of interaction messages, simplest option is to turn into 'role': 'message' block of text
        if isinstance(prompt, ChatPrompt):
            # This isn't used by either benchmark.
            chat_interactions_to_annotate = flatten_conversation_to_str(prompt)
        elif isinstance(prompt, TextPrompt):
            assert prompt.text is not None, "TextPrompt must have text."
            assert response.text is not None, "SUTResponse must have text."
            chat_interactions_to_annotate = f"User: {prompt.text}\n\nAssistant: {response.text}"
        else:
            raise ValueError(f"Can only handle ChatPrompt, TextPrompt types. Type given: {type(prompt)}")

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

    @staticmethod
    def _parse_text_to_safety_model_response(response_text: str) -> SafetyAnnotation:
        """Parse the response text into a SafetyModelResponse object."""
        json_marker = "{"
        try:
            start_index = response_text.index(json_marker)
        except ValueError:
            return PromptEngineeredAnnotator._get_invalid_response(response_text)
        cleaned_str = response_text[start_index:].replace("\n", "")
        end_index = cleaned_str.rfind("}")
        if end_index != -1:
            cleaned_str = cleaned_str[: end_index + 1]
        metadata = {}
        try:
            metadata = json.loads(cleaned_str)
            is_safe = metadata.get("is_safe")
            del metadata["is_safe"]
            if "is_valid" in metadata:
                del metadata["is_valid"]
        except json.JSONDecodeError as e:
            # 2. Try to use a regex to extract "is_safe" only
            match_is_safe = _REGEX_IS_SAFE.search(response_text or "")
            if match_is_safe:
                is_safe = match_is_safe.group(1).lower() == "true"
            else:
                return PromptEngineeredAnnotator._get_invalid_response(response_text)

        # add required fields as needed
        is_valid = True
        if (
            "safety_categories" not in metadata
            or not isinstance(metadata["safety_categories"], list)
            or any(not isinstance(i, str) for i in metadata["safety_categories"])
        ):
            metadata["safety_categories"] = []
        if "raw_response" not in metadata:
            metadata["raw_response"] = response_text

        return SafetyAnnotation(is_safe=is_safe, is_valid=is_valid, metadata=metadata)

    @staticmethod
    def _get_invalid_response(response_text: str) -> SafetyAnnotation:
        return SafetyAnnotation(
            is_safe=False,
            is_valid=False,
            metadata={"raw_response": response_text, "safety_categories": []},
        )

    def translate_response(self, request, response) -> SafetyAnnotation:
        """Convert the raw response into the form read by Tests."""
        translated_response: SUTResponse = self.model.translate_response(request, response)
        response_text = translated_response.text

        try:
            return self._parse_text_to_safety_model_response(response_text)
        except Exception:
            logger.exception(f"Error parsing response JSON", exc_info=True)
            return PromptEngineeredAnnotator._get_invalid_response(response_text)


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
