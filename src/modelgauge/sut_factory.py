from enum import Enum

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, UnknownSUTMakerError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_registry import SUTS
from modelgauge.suts.huggingface_sut_factory import HuggingFaceSUTFactory
from modelgauge.suts.openai_sut_factory import OpenAISUTFactory
from modelgauge.suts.together_sut_factory import TogetherSUTFactory


class SUTNotFoundException(Exception):
    pass


class SUTType(Enum):
    DYNAMIC = "dynamic"
    KNOWN = "known"
    UNKNOWN = "unknown"


# TODO: Auto-collect. Maybe make "driver" keys a constant in each factory module.
# Maps a string to the module and factory function in that module
# that can be used to create a dynamic sut
DYNAMIC_SUT_FACTORIES: dict = {
    "proxied": {"hfrelay": HuggingFaceSUTFactory},
    "direct": {
        "openai": OpenAISUTFactory,
        "together": TogetherSUTFactory,
        "huggingface": HuggingFaceSUTFactory,
        "hf": HuggingFaceSUTFactory,
    },
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
}


class SUTFactory:
    """A factory for both pre-registered and dynamic SUTs."""

    def __init__(self, sut_registry):
        self.sut_registry = sut_registry
        self.dynamic_sut_factories = self._load_dynamic_sut_factories(load_secrets_from_config())

    def _load_dynamic_sut_factories(self, secrets: RawSecrets) -> dict[str, dict[str, DynamicSUTFactory]]:
        factories: dict[str, dict[str, DynamicSUTFactory]] = {"direct": {}, "proxied": {}}
        for driver, factory_class in DYNAMIC_SUT_FACTORIES["direct"].items():
            factories["direct"][driver] = factory_class(secrets)
        for driver, factory_class in DYNAMIC_SUT_FACTORIES["proxied"].items():
            factories["proxied"][driver] = factory_class(secrets)
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
        sut_metadata: DynamicSUTMetadata = DynamicSUTMetadata.parse_sut_uid(uid)

        if sut_metadata.is_proxied():
            factory = self.dynamic_sut_factories["proxied"].get(sut_metadata.driver, None)
        else:
            factory = self.dynamic_sut_factories["direct"].get(sut_metadata.driver, None)  # type: ignore
        if not factory:
            raise UnknownSUTMakerError(f'Don\'t know how to make dynamic sut "{uid}"')
        return factory.make_sut(sut_metadata)

    def keys(self) -> list[str]:
        """Mimic the registry interface."""
        return self.sut_registry.keys()

    def get_missing_dependencies(self, uid: str, *, secrets: RawSecrets):
        """Mimic the registry interface. Only obtain missing secrets for PRE-REGISTERED SUTs."""
        if self._classify_sut_uid(uid) == SUTType.DYNAMIC:
            return []
        return self.sut_registry.get_missing_dependencies(uid, secrets=secrets)


SUT_FACTORY = SUTFactory(SUTS)
