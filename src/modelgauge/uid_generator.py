from collections import defaultdict
from typing import Optional


from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.sut_specification import SUTSpecification, SUTSpecificationChunk


class UID(str):
    def __init__(self, value):
        self = value

    @staticmethod
    def is_valid_uid(uid: str) -> bool:
        return False  # todo


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
            self._generate()
        return self._uid

    def _generate(self):
        pass

    def add(self, key, value):
        if self.data.get(key, None) == value:  # already added
            return
        if isinstance(value, str):
            value = value.strip()
        self.data[key] = value
        self._fresh = False


class SUTUIDGenerator(UIDGenerator):
    spec = SUTSpecification()
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

    @staticmethod
    def kv_to_str(field, value) -> str:
        if isinstance(value, str):
            value = value.replace(" ", SUTUIDGenerator.space_sub)
        return f"{field}{SUTUIDGenerator.key_value_separator}{value}"

    @staticmethod
    def bool_to_str(value):
        return "y" if value else "n"

    def validate(self):
        for key, field in self.spec.fields.items():
            value = self.data.get(field.name, None)
            if field.required and value is None:
                raise ValueError(f"Field {field.name} is required.")
            if value is not None and not isinstance(value, field.type):
                raise ValueError(f"Field {field.name} has wrong type.")

    def _generate(self):
        self.validate()
        chunks = []

        # the first chunks follow the dynamic SUT schema
        metadata: DynamicSUTMetadata = self.to_dynamic_sut_metadata()
        chunks.append(str(metadata))

        for field in SUTUIDGenerator.order:
            value = self.data.get(field, None)
            label = SUTUIDGenerator.spec.fields[field].label
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
