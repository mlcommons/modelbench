import os

import together  # type: ignore
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret
from modelgauge.suts.together_client import TogetherApiKey, TogetherChatSUT


DRIVER_NAME = "together"


class TogetherSUTFactory(DynamicSUTFactory):

    @staticmethod
    def get_secrets() -> InjectSecret:
        api_key = InjectSecret(TogetherApiKey)
        return api_key

    @staticmethod
    def find(sut_metadata: DynamicSUTMetadata):
        clean_up = False
        env_key = os.environ.get("TOGETHER_API_KEY", None)

        if not env_key:
            secrets = load_secrets_from_config()
            env_key = TogetherApiKey.make(secrets).value
            os.environ["TOGETHER_API_KEY"] = env_key
            clean_up = True

        found = None

        try:
            model_list = together.Models.list()
            found = [
                model["id"] for model in model_list if model["id"].lower() == sut_metadata.external_model_name().lower()
            ][0]
        except Exception as e:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together: {e}"
            )

        if clean_up:
            del os.environ["TOGETHER_API_KEY"]

        return found

    @staticmethod
    def make_sut(sut_metadata: DynamicSUTMetadata):
        model_name = TogetherSUTFactory.find(sut_metadata)
        if not model_name:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together."
            )

        assert sut_metadata.driver == DRIVER_NAME
        return (
            TogetherChatSUT,
            DynamicSUTMetadata.make_sut_uid(sut_metadata),
            sut_metadata.external_model_name(),
            TogetherSUTFactory.get_secrets(),
        )
