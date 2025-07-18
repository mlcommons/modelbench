from together import Together  # type: ignore

from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.suts.together_client import TogetherChatSUT


DRIVER_NAME = "together"


class TogetherSUTFactory(DynamicSUTFactory):
    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None  # Lazy load.

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    @staticmethod
    def get_secrets() -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        return [api_key]

    def _find(self, sut_metadata: DynamicSUTMetadata):
        found = None
        try:
            model_list = self.client.Models.list()
            found = [
                model["id"] for model in model_list if model["id"].lower() == sut_metadata.external_model_name().lower()
            ][0]
        except Exception as e:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together: {e}"
            )

        return found

    def make_sut(self, sut_metadata: DynamicSUTMetadata) -> TogetherChatSUT:
        model_name = self._find(sut_metadata)
        if not model_name:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together."
            )

        assert sut_metadata.driver == DRIVER_NAME
        return TogetherChatSUT(
            DynamicSUTMetadata.make_sut_uid(sut_metadata),
            sut_metadata.external_model_name(),
            *self.injected_secrets(),
        )
