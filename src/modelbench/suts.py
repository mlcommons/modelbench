import dataclasses
import functools

from modelgauge.sut_registry import SUTS


@dataclasses.dataclass
class SutDescription:
    key: str

    @property
    def uid(self):
        return self.key


@dataclasses.dataclass
class ModelGaugeSut(SutDescription):
    @classmethod
    @functools.cache
    def for_key(cls, key: str) -> "ModelGaugeSut":
        valid_keys = [item[0] for item in SUTS.items()]
        if key in valid_keys:
            return ModelGaugeSut(key)
        else:
            raise ValueError(f"Unknown SUT {key}; valid keys are {valid_keys}")

    def __hash__(self):
        return self.key.__hash__()

    def instance(self, secrets):
        if not hasattr(self, "_instance"):
            if secrets is None:
                return None
            self._instance = SUTS.make_instance(self.key, secrets=secrets)
        return self._instance

    def instance_initialization(self):
        instance = self.instance(None)
        if instance:
            return instance.initialization_record
