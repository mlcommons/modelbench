import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Union

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
        self._fields = {
            "model": SUTSpecificationElement("model", "m", str, True),
            "driver": SUTSpecificationElement("driver", "d", str, True),
            "maker": SUTSpecificationElement("maker", "mk", str),
            "provider": SUTSpecificationElement("provider", "pr", str),
            "display_name": SUTSpecificationElement("display_name", "dn", str),
            "reasoning": SUTSpecificationElement("reasoning", "reas", bool),
            "moderated": SUTSpecificationElement("moderated", "mod", bool),
            "date": SUTSpecificationElement("date", "dt", str),
            "base_url": SUTSpecificationElement("base_url", "url", str),
        }
        self._fields_by_label = {v.label: v for (_, v) in self._fields.items()}

    def knows(self, field):
        return field in self._fields

    def requires(self, field):
        return self.knows(field) and self._fields[field].required

    def validate(self, data: dict) -> bool:
        for field_spec in self._fields.values():
            value = data.get(field_spec.name, None)
            if field_spec.required and value is None:
                raise ValueError(f"Field {field_spec.name} is required.")
            if value is not None and not isinstance(value, field_spec.value_type):
                raise ValueError(f"Field {field_spec.name} has wrong type {type(value)}.")
        return True

    def label(self, field: str):
        return self._fields[field].label

    def element_for_label(self, label: str):
        return self._fields_by_label[label]


DEFINITION_VALUE_TYPES = Union[str, int, float, bool, None]


class SUTDefinition:
    """The data in a SUT configuration file or JSON blob"""

    _data: dict[str, DEFINITION_VALUE_TYPES]

    def __init__(self, data=None, **kwargs):
        self._uid: str = ""
        self._dynamic_uid: str = ""
        self._frozen = False
        self.spec = SUTSpecification()
        self._data = {}
        if data:
            for k, v in data.items():
                self.add(k, v)
        for k, v in kwargs.items():
            self.add(k, v)

    @staticmethod
    def from_arg(input: str) -> "SUTDefinition | None":
        try:
            sut_definition = SUTDefinition.from_json(input)
            sut_definition.validate()
            return sut_definition
        except:
            if SUTUIDGenerator.is_rich_sut_uid(input):
                try:
                    sut_definition = SUTUIDGenerator.parse(input)
                    sut_definition.validate()
                    return sut_definition
                except:
                    return None
        return None

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
    def canonicalize(sut_uid: str) -> str:
        sd = SUTDefinition.from_arg(sut_uid)
        if sd:
            return sd.uid
        else:
            return sut_uid

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

    def __str__(self):
        return self.uid

    def _generate_uids(self):
        generator = SUTUIDGenerator(self)
        self._uid = generator.uid
        self._dynamic_uid = generator._generate_dynamic_uid()
        self._frozen = True

    def validate(self) -> bool:
        return self.spec.validate(self._data)

    def add(self, key: str, value: DEFINITION_VALUE_TYPES):
        if self._frozen:
            raise AttributeError(f"Attempting to add item {key} to a frozen SUTDefinition.")
        if isinstance(value, str):
            value = value.strip()
        if self.spec.knows(key):
            self._data[key] = value
        else:
            raise ValueError(f"Don't know what to do with {key}")

    def add_sut_metadata(self, metadata: DynamicSUTMetadata):
        self.add("model", metadata.model)
        self.add("maker", metadata.maker)
        self.add("driver", metadata.driver)
        self.add("provider", metadata.provider)
        self.add("date", metadata.date)

    def get(self, field: str, default=None) -> DEFINITION_VALUE_TYPES:
        return self._data.get(field, default)

    def to_dynamic_sut_metadata(self) -> DynamicSUTMetadata:
        return DynamicSUTMetadata(
            model=self._data["model"],
            driver=self._data["driver"],
            maker=self._data.get("maker", None),
            provider=self._data.get("provider", None),
            date=self._data.get("date", None),
        )

    def external_model_name(self) -> str:
        metadata = self.to_dynamic_sut_metadata()
        return metadata.external_model_name()


class SUTUIDGenerator:
    # This is the order past the dynamic SUT UID fields, which are fixed and at the head of this UID
    # It's arbitrary and made up pending a group decision
    order = (
        "moderated",
        "reasoning",
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
            label = self.definition.spec.label(field)
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
            label, value = bits
            the_element = definition.spec.element_for_label(label)
            if not the_element:
                warnings.warn(f"Unknown chunk {label} found in {uid}")
                continue
            if the_element.value_type in (int, float):
                value = the_element.value_type(value)
            elif the_element.value_type is bool:
                value = SUTUIDGenerator.str_to_bool(value)
            elif the_element.value_type is str:
                value = value.replace(SUTUIDGenerator.space_sub, " ")
            definition.add(the_element.name, value)

        return definition
