from collections import defaultdict
from typing import Optional

from pydantic import BaseModel

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata


class UID(str):
    def __init__(self, value):
        self = value

    @staticmethod
    def is_valid_uid(uid: str) -> bool:
        return False  # todo


class UIDChunk(BaseModel):
    name: str
    label: str
    type: type
    required: Optional[bool] = False

    @staticmethod  # sub for pydantic's verbose constructor
    def make(name, label, type, required: bool = False):
        return UIDChunk(name=name, label=label, type=type)


class UIDGenerator:

    def __init__(self, data=None):
        self.data = defaultdict()
        if data:
            for k, v in data.items():
                self.add(k, v)
        self._uid: str = ""
        self._fresh = False

    @property
    def uid(self) -> str:
        if not self._fresh:
            self.__generate()
        return self._uid

    def __generate(self):
        pass

    def add(self, key, value):
        if self.data.get(key, None) == value:  # already added
            return
        if isinstance(value, str):
            value = value.strip()
        self.data[key] = value
        self._fresh = False


class SUTUIDGenerator(UIDGenerator):
    fields = {
        "model": UIDChunk.make("model", "m", str, True),
        "temperature": UIDChunk.make("temperature", "t", float),
        "top_p": UIDChunk.make("top_p", "p", int),
        "top_k": UIDChunk.make("top_k", "k", int),
        "driver": UIDChunk.make("driver", "d", str, True),
        "maker": UIDChunk.make("maker", "mk", str),
        "provider": UIDChunk.make("provider", "pr", str),
        "display_name": UIDChunk.make("display_name", "dn", str),
        "reasoning": UIDChunk.make("reasoning", "reas", bool),
        "moderated": UIDChunk.make("moderated", "mod", bool),
        "driver_code_version": UIDChunk.make("driver_code_version", "dv", str),
        "date": UIDChunk.make("date", "dt", str),
    }
    order = (
        "driver_code_version",
        "moderated",
        "reasoning",
        "temperature",
        "top_p",
        "top_k",
        "display_name",
    )  # this is the order past the dynamic SUT UID fields, which are fixed and at the head of this UID
    field_separator = "_"
    key_value_separator = ":"
    blank_sub = "."
    space_sub = "-"

    @property
    def uid(self) -> str:
        if not self._fresh:
            self.__generate()
        return self._uid

    @staticmethod
    def kv_to_str(field, value) -> str:
        if isinstance(value, str):
            value = value.replace(" ", SUTUIDGenerator.space_sub)
        return f"{field}{SUTUIDGenerator.key_value_separator}{value}"

    @staticmethod
    def bool_to_str(value):
        return "y" if value else "n"

    def validate(self):
        for key, field in self.fields.items():
            value = self.data.get(field.name, None)
            if field.required and value is None:
                raise ValueError(f"Field {field.name} is required.")
            if value is not None and not isinstance(value, field.type):
                raise ValueError(f"Field {field.name} has wrong type.")

    def __generate(self):
        self.validate()
        chunks = []

        # the first chunks follow the dynamic SUT schema
        metadata: DynamicSUTMetadata = self.to_dynamic_sut_metadata()
        chunks.append(str(metadata))

        for field in SUTUIDGenerator.order:
            value = self.data.get(field, None)
            label = SUTUIDGenerator.fields[field].label
            if isinstance(value, bool):
                value = SUTUIDGenerator.bool_to_str(value)
            if value:
                chunks.append(SUTUIDGenerator.kv_to_str(label, value))

        self._uid = SUTUIDGenerator.field_separator.join(chunks).lower()
        self._fresh = True
        return self._uid

    def to_dynamic_sut_metadata(self) -> DynamicSUTMetadata:
        return DynamicSUTMetadata(
            model=self.data["model"],
            driver=self.data["driver"],
            maker=self.data.get("maker", None),
            provider=self.data.get("provider", None),
            date=self.data.get("date", None),
        )
