from dataclasses import dataclass
from pprint import pprint
from typing import Optional
import json
import os
import warnings

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata, RICH_UID_FIELD_SEPARATOR, SEPARATOR


@dataclass
class SUTSpecificationElement:
    name: str = ""
    label: str = ""
    value_type: type = str
    required: Optional[bool] = False


class SUTSpecification:
    """The spec a SUT definition needs to comply with"""

    def __init__(self):
        self.fields = {
            "model": SUTSpecificationElement("model", "m", str, True),
            "driver": SUTSpecificationElement("driver", "d", str, True),
            "temp": SUTSpecificationElement("temp", "t", float),
            "max_tokens": SUTSpecificationElement("max_tokens", "mt", int),
            "top_p": SUTSpecificationElement("top_p", "p", int),
            "top_k": SUTSpecificationElement("top_k", "k", int),
            "top_logprobs": SUTSpecificationElement("top_logprobs", "l", int),
            "maker": SUTSpecificationElement("maker", "mk", str),
            "provider": SUTSpecificationElement("provider", "pr", str),
            "display_name": SUTSpecificationElement("display_name", "dn", str),
            "reasoning": SUTSpecificationElement("reasoning", "reas", bool),
            "moderated": SUTSpecificationElement("moderated", "mod", bool),
            "driver_code_version": SUTSpecificationElement("driver_code_version", "dv", str),
            "date": SUTSpecificationElement("date", "dt", str),
            "base_url": SUTSpecificationElement("base_url", "url", str),
        }

    def knows(self, field):
        return field in self.fields

    def requires(self, field):
        return self.knows(field) and self.fields[field].required

    def validate(self, data: dict) -> bool:
        for field_spec in self.fields.values():
            value = data.get(field_spec.name, None)
            if field_spec.required and value is None:
                raise ValueError(f"Field {field_spec.name} is required.")
            if value is not None and not isinstance(value, field_spec.value_type):
                raise ValueError(f"Field {field_spec.name} has wrong type {type(value)}.")
        return True


class SUTDefinition:
    """The data in a SUT configuration file or JSON blob"""

    def __init__(self, data=None, **kwargs):
        self._uid: str = ""
        self._dynamic_uid: str = ""
        self._frozen = False
        self.spec = SUTSpecification()
        self.data = {}
        if data:
            for k, v in data.items():
                self.add(k, v)
        for k, v in kwargs.items():
            self.add(k, v)

    @staticmethod
    def from_json(data: str):
        """Makes a SUTDefinition based on either a string of JSON or a file containing that JSON"""
        if SUTUIDGenerator.is_json_string(data):
            return SUTDefinition.from_json_string(data)
        elif SUTUIDGenerator.is_file(data):
            return SUTDefinition.from_json_file(data)
        else:
            raise ValueError(f"Unable to do anything with {data}")

    @staticmethod
    def from_json_string(data: str) -> "SUTDefinition":
        try:
            data = json.loads(data)
            definition = SUTDefinition(data)
            return definition
        except:
            raise ValueError(f"Malformed json input: {data}")

    @staticmethod
    def from_json_file(path: str) -> "SUTDefinition":
        try:
            f = open(path, "r")
            data = json.load(f)
            definition = SUTDefinition(data)
            return definition
        except:
            raise OSError(f"Unable to read data from {path}")

    @property
    def uid(self):
        if not self._frozen:
            self._generate_uids()
        return self._uid

    @property
    def dynamic_uid(self):
        if not self._frozen:
            self._generate_uids()
        return self._dynamic_uid

    # TODO: is this handy?
    def __str__(self):
        return self.uid

    def _generate_uids(self):
        generator = SUTUIDGenerator(self)
        self._uid = generator.uid
        self._dynamic_uid = generator._generate_dynamic_uid()
        self._frozen = True

    def validate(self) -> bool:
        return self.spec.validate(self.data)

    def add(self, key, value):
        if self._frozen:
            raise AttributeError(f"Attempting to add item {key} to a frozen SUTDefinition.")
        if isinstance(value, str):
            value = value.strip()
        if self.spec.knows(key):
            self.data[key] = value
        else:
            raise ValueError(f"Don't know what to do with {key}")

    def add_sut_metadata(self, metadata):
        self.add("model", metadata.model)
        self.add("maker", metadata.maker)
        self.add("driver", metadata.driver)
        self.add("provider", metadata.provider)
        self.add("date", metadata.date)

    def get(self, field, default=None):
        return self.data.get(field, default)

    def dump(self):
        pprint(self.data, indent=4)

    def to_dynamic_sut_metadata(self) -> DynamicSUTMetadata:
        return DynamicSUTMetadata(
            model=self.data["model"],
            driver=self.data["driver"],
            maker=self.data.get("maker", None),
            provider=self.data.get("provider", None),
            date=self.data.get("date", None),
        )

    def external_model_name(self):
        metadata = self.to_dynamic_sut_metadata()
        return metadata.external_model_name()


class SUTUIDGenerator:
    # This is the order past the dynamic SUT UID fields, which are fixed and at the head of this UID
    # It's arbitrary and made up pending a group decision
    order = (
        "driver_code_version",
        "moderated",
        "reasoning",
        "max_tokens",
        "temp",
        "top_p",
        "top_k",
        "top_logprobs",
        "display_name",
        "base_url",  # for OpenAI-compatible SUTs
    )
    field_separator = RICH_UID_FIELD_SEPARATOR
    key_value_separator = "="
    blank_sub = "."
    space_sub = "_"

    def __init__(self, definition: SUTDefinition):
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

    @staticmethod
    def str_to_bool(value):
        return value == "y"

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

    def _generate_dynamic_uid(self) -> str:
        chunks = []
        metadata: DynamicSUTMetadata = self.definition.to_dynamic_sut_metadata()
        chunks.append(str(metadata))
        uid = SUTUIDGenerator.field_separator.join(chunks).lower()
        return uid

    @staticmethod
    def is_rich_sut_uid(uid):
        if SUTUIDGenerator.is_json_string(uid):
            return False
        return SUTUIDGenerator.field_separator in uid or SUTUIDGenerator.key_value_separator in uid or SEPARATOR in uid

    @staticmethod
    def is_json_string(uid):
        it_is_json = False
        try:
            _ = json.loads(uid)
            it_is_json = True
        except:
            pass
        return it_is_json

    @staticmethod
    def is_file(uid):
        return not SUTUIDGenerator.is_json_string(uid) and os.path.exists(uid)

    @staticmethod
    def parse(uid: str) -> SUTDefinition:
        definition = SUTDefinition()
        reversed = {}
        for element in definition.spec.fields.values():
            reversed[element.label] = element

        chunks = uid.split(SUTUIDGenerator.field_separator, 1)
        if len(chunks) > 1:
            dynamic_uid, sut_options = chunks
        else:
            dynamic_uid = chunks[0]
            sut_options = ""

        metadata = DynamicSUTMetadata.parse_sut_uid(dynamic_uid)
        definition.add_sut_metadata(metadata)

        for chunk in sut_options.split(SUTUIDGenerator.field_separator):
            if not chunk:
                continue
            bits = chunk.split(SUTUIDGenerator.key_value_separator)
            param, value = bits
            the_element = reversed.get(param, None)
            if not the_element:
                warnings.warn(f"Unknown chunk {param} found in {uid}")
                continue
            if the_element.value_type in (int, float):
                value = the_element.value_type(value)
            elif the_element.value_type is bool:
                value = SUTUIDGenerator.str_to_bool(value)
            elif the_element.value_type is str:
                value = value.replace(SUTUIDGenerator.space_sub, " ")
            definition.add(the_element.name, value)

        return definition
