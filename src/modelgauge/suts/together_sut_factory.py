import os

import together  # type: ignore
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret
from modelgauge.suts.together_client import TogetherApiKey, TogetherChatSUT


class TogetherSUTFactory(DynamicSUTFactory):

    @staticmethod
    def get_secrets() -> InjectSecret:
        api_key = InjectSecret(TogetherApiKey)
        return api_key

    @staticmethod
    def find(name: str):
        clean_up = False
        env_key = os.environ.get("TOGETHER_API_KEY", None)

        if not env_key:
            secrets = load_secrets_from_config()
            env_key = TogetherApiKey.make(secrets).value
            os.environ["TOGETHER_API_KEY"] = env_key
            clean_up = True

        found = None

        try:
            metadata = DynamicSUTMetadata.parse_sut_uid(name)
            model_list = together.Models.list()
            found = [
                model["id"] for model in model_list if model["id"].lower() == metadata.external_model_name().lower()
            ][0]
        except Exception as e:
            raise ModelNotSupportedError(f"Model {name} not found or not available on together: {e}")

        if clean_up:
            del os.environ["TOGETHER_API_KEY"]

        return found

    @staticmethod
    def make_sut(name: str):
        model_name = TogetherSUTFactory.find(name)
        if not model_name:
            raise ModelNotSupportedError(f"Model {name} not found or not available on together.")

        metadata = DynamicSUTMetadata.parse_sut_uid(name)
        assert metadata.provider == "together"
        return (
            TogetherChatSUT,
            DynamicSUTMetadata.make_sut_uid(metadata),
            metadata.external_model_name(),
            TogetherSUTFactory.get_secrets(),
        )
