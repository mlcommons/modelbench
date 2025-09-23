import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Union, Mapping

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata, RICH_UID_FIELD_SEPARATOR, SEPARATOR


@dataclass
class SUTSpecificationElement:
    name: str = ""
    label: str = ""
    value_type: type = str
    required: Optional[bool] = False

    def matches(self, field_name):
        return field_name == self.name

    def name_for_label(self, label):
        if self.label == label:
            return self.name
        raise (ValueError(f"for static elements, {label} must match {self.label}"))


class PrefixSUTSpecificationElement(SUTSpecificationElement):

    def matches(self, field_name):
        return field_name.startswith(self.name)

    def name_for_label(self, label):
        return label


class SUTSpecification:
    """The spec a SUT definition needs to comply with"""

    def __init__(self):
        fields = [
            SUTSpecificationElement("model", "m", str, True),
            SUTSpecificationElement("driver", "d", str, True),
            SUTSpecificationElement("maker", "mk", str),
            SUTSpecificationElement("provider", "pr", str),
            SUTSpecificationElement("display_name", "dn", str),
            SUTSpecificationElement("reasoning", "reas", bool),
            SUTSpecificationElement("moderated", "mod", bool),
            SUTSpecificationElement("date", "dt", str),
            SUTSpecificationElement("base_url", "url", str),
        ]

        self._wildcard_fields = [PrefixSUTSpecificationElement("vllm-", "vllm", str)]

        self._fields_by_name = {v.name: v for v in fields}
        self._fields_by_label = {v.label: v for v in fields}

    def knows(self, name: str):
        return name in self._fields_by_name or any([f.matches(name) for f in self._wildcard_fields])

    def requires(self, name: str):
        return self.knows(name) and self._fields_by_name[name].required

    def validate(self, data: dict) -> bool:
        for field_spec in self._fields_by_name.values():
            value = data.get(field_spec.name, None)
            if field_spec.required and value is None:
                raise ValueError(f"Field {field_spec.name} is required.")
            if value is not None and not isinstance(value, field_spec.value_type):
                raise ValueError(f"Field {field_spec.name} has wrong type {type(value)}.")
        return True

    def label(self, name: str):
        return self._fields_by_name[name].label

    def element_for_label(self, label: str):
        if label in self._fields_by_label:
            return self._fields_by_label[label]
        for element in self._wildcard_fields:
            if element.matches(label):
                return element
        return None


DEFINITION_VALUE_TYPES = Union[str, int, float, bool, None]


class SUTDefinition:
    """The data in a SUT configuration file or JSON blob"""

    _data: dict[str, DEFINITION_VALUE_TYPES]

    def __init__(self, data=None, **kwargs):
        self.spec = SUTSpecification()
        self._data = {}

        if data:
            for k, v in data.items():
                self._add(k, v)
        for k, v in kwargs.items():
            self._add(k, v)
        if not self.spec.validate(self._data):
            raise ValueError(f"Invalid data: {self._data}")

        generator = SUTUIDGenerator(self)
        self.uid = generator.uid
        self.dynamic_uid = generator._generate_dynamic_uid()

    def __str__(self):
        return self.uid

    def _add(self, key: str, value: DEFINITION_VALUE_TYPES):
        if isinstance(value, str):
            value = value.strip()
        if self.spec.knows(key):
            self._data[key] = value
        else:
            raise ValueError(f"Don't know what to do with {key}")

    def get(self, field: str, default=None) -> DEFINITION_VALUE_TYPES:
        return self._data.get(field, default)

    def get_matching(self, label: str) -> Mapping[str, DEFINITION_VALUE_TYPES] | None:
        element = self.spec.element_for_label(label)
        if not element:
            return None
        result = {}
        for k, v in self._data.items():
            if element.matches(k):
                result[k] = v
        return result

    def to_dynamic_sut_metadata(self) -> DynamicSUTMetadata:
        return DynamicSUTMetadata(
            model=self._data["model"],  # type: ignore
            driver=self._data["driver"],  # type: ignore
            maker=self._data.get("maker", None),  # type: ignore
            provider=self._data.get("provider", None),  # type: ignore
            date=self._data.get("date", None),  # type: ignore
        )

    def external_model_name(self) -> str:
        metadata = self.to_dynamic_sut_metadata()
        return metadata.external_model_name()

    @staticmethod
    def from_arg(input: str) -> "SUTDefinition | None":
        try:
            return SUTDefinition.from_json(input)
        except:
            if SUTUIDGenerator.is_rich_sut_uid(input):
                try:
                    return SUTDefinition.parse(input)
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

    @staticmethod
    def parse(uid: str) -> "SUTDefinition":
        spec = SUTSpecification()

        chunks = uid.split(SUTUIDGenerator.field_separator, 1)
        if len(chunks) > 1:
            dynamic_uid, sut_options = chunks
        else:
            dynamic_uid = chunks[0]
            sut_options = ""

        metadata = DynamicSUTMetadata.parse_sut_uid(dynamic_uid)
        data = metadata.model_dump()

        for chunk in sut_options.split(SUTUIDGenerator.field_separator):
            if not chunk:
                continue
            label, value = chunk.split(SUTUIDGenerator.key_value_separator)
            the_element = spec.element_for_label(label)
            if not the_element:
                warnings.warn(f"Unknown chunk {label} found in {uid}")
                continue
            if the_element.value_type in (int, float):
                value = the_element.value_type(value)
            elif the_element.value_type is bool:
                value = SUTUIDGenerator.str_to_bool(value)
            elif the_element.value_type is str:
                value = value.replace(SUTUIDGenerator.space_sub, " ")
            data[the_element.name_for_label(label)] = value

        definition = SUTDefinition(data)

        return definition


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
