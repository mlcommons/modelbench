from enum import Enum

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, UnknownSUTMakerError
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition, SUTUIDGenerator
from modelgauge.sut_registry import SUTS
from modelgauge.suts.huggingface_sut_factory import HuggingFaceSUTFactory
from modelgauge.suts.modelship_sut import ModelShipSUTFactory
from modelgauge.suts.openai_sut_factory import OpenAICompatibleSUTFactory
from modelgauge.suts.together_sut_factory import TogetherSUTFactory


class SUTNotFoundException(Exception):
    pass


class SUTType(Enum):
    DYNAMIC = "dynamic"
    KNOWN = "known"
    UNKNOWN = "unknown"


# TODO: Auto-collect?
# Make sure the factory module includes the matching key as a constant.
# Maps a string to the module and factory function in that module
# that can be used to create a dynamic sut
DYNAMIC_SUT_FACTORIES: dict = {
    "hf": HuggingFaceSUTFactory,
    "hfrelay": HuggingFaceSUTFactory,
    "huggingface": HuggingFaceSUTFactory,
    "openai": OpenAICompatibleSUTFactory,
    "together": TogetherSUTFactory,
    "modelship": ModelShipSUTFactory,
}

LEGACY_SUT_MODULE_MAP = {
    # HuggingFaceChatCompletionDedicatedSUT and HuggingFaceChatCompletionServerlessSUT
    "nvidia-llama-3-1-nemotron-nano-8b-v1": "huggingface_chat_completion",
    "athene-v2-chat-hf": "huggingface_chat_completion",
    "aya-expanse-8b-hf": "huggingface_chat_completion",
    "gemma-2-9b-it-hf": "huggingface_chat_completion",
    "gemma-2-9b-it-simpo-hf": "huggingface_chat_completion",
    "granite-3-1-8b-instruct-hf": "huggingface_chat_completion",
    "llama-3-1-tulu-3-8b-hf": "huggingface_chat_completion",
    "llama-3-1-tulu-3-70b-hf": "huggingface_chat_completion",
    "mistral-nemo-instruct-2407-hf": "huggingface_chat_completion",
    "mixtral-8x22b-instruct-v0-1-hf": "huggingface_chat_completion",
    "olmo-2-1124-13b-instruct-hf": "huggingface_chat_completion",
    "olmo-2-0325-32b-instruct-hf": "huggingface_chat_completion",
    "qwen1-5-110b-chat-hf": "huggingface_chat_completion",
    "qwen2-5-7b-instruct-hf": "huggingface_chat_completion",
    "qwq-32b-hf": "huggingface_chat_completion",
    "yi-1-5-34b-chat-hf": "huggingface_chat_completion",
    "cohere-c4ai-command-a-03-2025-hf": "huggingface_chat_completion",
    "meta-llama-3_1-8b-instruct-hf-nebius": "huggingface_chat_completion",
    "google-gemma-3-12b-it-hf-featherless-ai": "huggingface_chat_completion",
    "google-gemma-3-27b-it-hf-nebius": "huggingface_chat_completion",
    # OpenAIChat
    "gpt-3.5-turbo": "openai_client",
    "gpt-4o": "openai_client",
    "gpt-4o-20250508": "openai_client",
    "gpt-4o-mini": "openai_client",
    # TogetherChatSUT and TogetherDedicatedChatSUT
    "llama-3-70b-chat": "together_client",
    "llama-3-70b-chat-hf": "together_client",
    "llama-3.1-8b-instruct-turbo-together": "together_client",
    "llama-3.1-405b-instruct-turbo-together": "together_client",
    "llama-3.3-70b-instruct-turbo-together": "together_client",
    "Mistral-7B-Instruct-v0.2": "together_client",
    "Mixtral-8x7B-Instruct-v0.1": "together_client",
    "mistral-8x22b-instruct": "together_client",
    "mistral-8x22b-instruct-nim": "together_client",
    "deepseek-R1": "together_client",
    "deepseek-v3-together": "together_client",
    "qwen2.5-7B-instruct-turbo-together": "together_client",
    "mistral-8x22b-instruct-dedicated-together": "together_client",
    # HuggingFaceSUT (endpoint-based)
    "olmo-7b-0724-instruct-hf": "huggingface_api",
    "olmo-2-1124-7b-instruct-hf": "huggingface_api",
    # MetaLlama
    "meta-llama-3.3-8b-instruct-llama": "meta_llama_client",
    "meta-llama-3.3-8b-instruct-moderated-llama": "meta_llama_client",
    # Demo SUTs
    "demo_yes_no": "demo_01_yes_no_sut",
    "demo_random_words": "demo_02_secrets_and_options_sut",
    "demo_always_angry": "demo_03_sut_with_args",
    "demo_always_sorry": "demo_03_sut_with_args",
    # AWS Bedrock
    "amazon-nova-1.0-micro": "aws_bedrock_client",
    "amazon-nova-1.0-lite": "aws_bedrock_client",
    "amazon-nova-1.0-pro": "aws_bedrock_client",
    "amazon-nova-1.0-premier": "aws_bedrock_client",
    # Anthropic
    "claude-3-5-sonnet-20241022": "anthropic_api",
    "claude-3-5-haiku-20241022": "anthropic_api",
    "claude-3-7-sonnet-20250219": "anthropic_api",
    # Azure
    "phi-3.5-mini-instruct": "azure_client",
    "phi-3.5-moe-instruct": "azure_client",
    "phi-4": "azure_client",
    "ai21labs-AI21-Jamba-Large-1.5-azure": "azure_client",
    # BaseTen
    "nvidia-llama-3.3-49b-nemotron-super": "baseten_api",
    # Google
    "google-genai-gemini-2.5-flash-preview-05-20-no-reasoning": "google_genai",
    "gemini-1.5-flash-8b": "google_generativeai",
    "gemini-1.5-pro": "google_generativeai",
    "gemini-2.0-flash": "google_generativeai",
    "gemini-2.0-flash-lite": "google_generativeai",
    "gemini-2.0-flash-001": "google_generativeai",
    "gemini-2.5-pro-preview-03-25": "google_generativeai",
    "gemini-2.5-pro-preview-05-06": "google_generativeai",
    "gemini-1.5-flash-safety_block_none": "google_generativeai",
    "gemini-1.5-flash-8b-safety_block_none": "google_generativeai",
    "gemini-1.5-pro-safety_block_none": "google_generativeai",
    "gemini-2.0-flash-safety_block_none": "google_generativeai",
    "gemini-2.0-flash-lite-safety_block_none": "google_generativeai",
    "gemini-2.0-flash-001-safety_block_none": "google_generativeai",
    "gemini-2.5-pro-preview-03-25-safety_block_none": "google_generativeai",
    "gemini-2.5-pro-preview-05-06-safety_block_none": "google_generativeai",
    "gemini-1.5-flash-safety_block_most": "google_generativeai",
    "gemini-1.5-flash-8b-safety_block_most": "google_generativeai",
    "gemini-1.5-pro-safety_block_most": "google_generativeai",
    "gemini-2.0-flash-safety_block_most": "google_generativeai",
    "gemini-2.0-flash-lite-safety_block_most": "google_generativeai",
    "gemini-2.0-flash-001-safety_block_most": "google_generativeai",
    "gemini-2.5-pro-preview-03-25-safety_block_most": "google_generativeai",
    "gemini-2.5-pro-preview-05-06-safety_block_most": "google_generativeai",
    # Mistral
    "mistralai-ministral-8b-2410": "mistral_sut",
    "mistralai-mistral-large-2411": "mistral_sut",
    "mistralai-mistral-large-2402": "mistral_sut",
    "mistralai-ministral-8b-2410-moderated": "mistral_sut",
    "mistralai-mistral-large-2411-moderated": "mistral_sut",
    "mistralai-mistral-large-2402-moderated": "mistral_sut",
    # Nvidia
    "nvidia-llama-3.1-nemotron-70b-instruct": "nvidia_nim_api_client",
    "nvidia-nemotron-4-340b-instruct": "nvidia_nim_api_client",
    "nvidia-mistral-nemo-minitron-8b-8k-instruct": "nvidia_nim_api_client",
    "nvidia-nemotron-mini-4b-instruct": "nvidia_nim_api_client",
    # Vertex
    "vertexai-mistral-large-2411": "vertexai_mistral_sut",
}


class SUTFactory:
    """A factory for both pre-registered and dynamic SUTs."""

    def __init__(self, sut_registry):
        self.sut_registry = sut_registry
        self.dynamic_sut_factories = self._load_dynamic_sut_factories(load_secrets_from_config())

    def _load_dynamic_sut_factories(self, secrets: RawSecrets) -> dict[str, DynamicSUTFactory]:
        factories: dict[str, DynamicSUTFactory] = {}
        for driver, factory_class in DYNAMIC_SUT_FACTORIES.items():
            factories[driver] = factory_class(secrets)
        return factories

    def knows(self, uid: str) -> bool:
        """Check if the registry knows about a given SUT UID. Dynamic SUTs are always considered known."""
        if self._classify_sut_uid(uid) == SUTType.DYNAMIC:
            return True
        return self.sut_registry.knows(uid)

    def make_instance(self, uid: str, *, secrets: RawSecrets) -> SUT:
        sut_type = self._classify_sut_uid(uid)
        if sut_type == SUTType.KNOWN:
            return self.sut_registry.make_instance(uid, secrets=secrets)
        elif sut_type == SUTType.DYNAMIC:
            return self._make_dynamic_sut(uid)
        else:
            raise SUTNotFoundException(f"{uid} is neither pre-registered nor a valid dynamic SUT UID.")

    def _classify_sut_uid(self, uid: str) -> SUTType:
        if uid in self.sut_registry.keys():
            return SUTType.KNOWN
        elif ":" in uid:
            return SUTType.DYNAMIC
        else:
            return SUTType.UNKNOWN

    def _make_dynamic_sut(self, uid: str) -> SUT:
        sut_definition: SUTDefinition = SUTUIDGenerator.parse(uid)
        factory = self.dynamic_sut_factories.get(sut_definition.get("driver"))  # type: ignore
        if not factory:
            raise UnknownSUTMakerError(f'Don\'t know how to make dynamic sut "{uid}"')
        return factory.make_sut(sut_definition)

    def keys(self) -> list[str]:
        """Mimic the registry interface."""
        return self.sut_registry.keys()

    def get_missing_dependencies(self, uid: str, *, secrets: RawSecrets):
        """Mimic the registry interface. Only obtain missing secrets for PRE-REGISTERED SUTs."""
        if self._classify_sut_uid(uid) == SUTType.DYNAMIC:
            return []
        return self.sut_registry.get_missing_dependencies(uid, secrets=secrets)


SUT_FACTORY = SUTFactory(SUTS)
