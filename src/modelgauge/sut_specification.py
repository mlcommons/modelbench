from collections import defaultdict
from typing import Optional
import json

from pydantic import BaseModel

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata


class SUTSpecificationElement(BaseModel):
    name: str = ""
    label: str = ""
    value_type: type = str
    required: Optional[bool] = False

    @staticmethod  # sub for pydantic's verbose constructor
    def make(name="", label="", value_type=type, required: bool = False):
        return SUTSpecificationElement(name=name, label=label, value_type=value_type, required=required)


class SUTSpecification:
    """The spec a SUT definition needs to comply with"""

    fields = {
        "model": SUTSpecificationElement.make("model", "m", str, True),
        "driver": SUTSpecificationElement.make("driver", "d", str, True),
        "temperature": SUTSpecificationElement.make("temperature", "t", float),
        "top_p": SUTSpecificationElement.make("top_p", "p", int),
        "top_k": SUTSpecificationElement.make("top_k", "k", int),
        "maker": SUTSpecificationElement.make("maker", "mk", str),
        "provider": SUTSpecificationElement.make("provider", "pr", str),
        "display_name": SUTSpecificationElement.make("display_name", "dn", str),
        "reasoning": SUTSpecificationElement.make("reasoning", "reas", bool),
        "moderated": SUTSpecificationElement.make("moderated", "mod", bool),
        "driver_code_version": SUTSpecificationElement.make("driver_code_version", "dv", str),
        "date": SUTSpecificationElement.make("date", "dt", str),
    }
    values = {}

    def knows(self, field):
        return field in self.fields

    def requires(self, field):
        return self.knows(field) and self.fields[field].required

    def validate(self, data: dict):
        for field in self.fields.values():
            value = data.get(field.name, None)
            if field.required and value is None:
                raise ValueError(f"Field {field.name} is required.")
            if value is not None and not isinstance(value, field.value_type):
                raise ValueError(f"Field {field.name} has wrong type.")


class SUTDefinition:
    """The data in a SUT configuration file or JSON blob"""

    spec: SUTSpecification = SUTSpecification()
    data: defaultdict = defaultdict(str)

    def __init__(self, data=None):
        if data:
            for k, v in data.items():
                self.add(k, v)
        self._uid: str = ""

    @staticmethod
    def from_json_string(data: str):
        try:
            data = json.loads(data)
            definition = SUTDefinition(data)
            return definition
        except:
            raise ValueError(f"Malformed json input: {data}")

    @staticmethod
    def from_json_file(path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                definition = SUTDefinition(data)
            return definition
        except Exception as exc:
            raise ValueError(f"Unable to read data from {path}")

    @property
    def uid(self):
        return self._generate_uid()

    def _generate_uid(self):
        generator = SUTUIDGenerator(self)
        return generator.uid

    def validate(self):
        return self.spec.validate(self.data)

    def add(self, key, value):
        if isinstance(value, str):
            value = value.strip()
        if self.spec.knows(key):
            self.data[key] = value
        else:
            raise ValueError(f"Don't know what to do with {key}")

    def get(self, field, default=None):
        return self.data.get(field, default)

    def to_dynamic_sut_metadata(self) -> DynamicSUTMetadata:
        return DynamicSUTMetadata(
            model=self.data["model"],
            driver=self.data["driver"],
            maker=self.data.get("maker", None),
            provider=self.data.get("provider", None),
            date=self.data.get("date", None),
        )


class SUTUIDGenerator:
    definition: SUTDefinition
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

    def __init__(self, definition: SUTDefinition | None = None):
        if definition:
            self.definition = definition
        self._uid: str = ""

    @staticmethod
    def kv_to_str(field, value) -> str:
        if isinstance(value, str):
            value = value.replace(" ", SUTUIDGenerator.space_sub)
        return f"{field}{SUTUIDGenerator.key_value_separator}{value}"

    @staticmethod
    def bool_to_str(value):
        return "y" if value else "n"

    @property
    def uid(self) -> str:
        return self._generate()

    def _generate(self):
        chunks = []

        # the first chunks follow the dynamic SUT schema
        metadata: DynamicSUTMetadata = self.definition.to_dynamic_sut_metadata()
        chunks.append(str(metadata))

        for field in SUTUIDGenerator.order:
            value = self.definition.get(field)
            label = self.definition.spec.fields[field].label
            if isinstance(value, bool):
                value = SUTUIDGenerator.bool_to_str(value)
            if value:
                chunks.append(SUTUIDGenerator.kv_to_str(label, value))

        self._uid = SUTUIDGenerator.field_separator.join(chunks).lower()
        return self._uid
