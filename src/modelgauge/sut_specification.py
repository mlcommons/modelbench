from typing import Optional

from pydantic import BaseModel


class SUTSpecificationChunk(BaseModel):
    name: str
    label: str
    type: type
    required: Optional[bool] = False

    @staticmethod  # sub for pydantic's verbose constructor
    def make(name, label, type, required: bool = False):
        return SUTSpecificationChunk(name=name, label=label, type=type, required=required)


class SUTSpecification:
    fields = {
        "model": SUTSpecificationChunk.make("model", "m", str, True),
        "temperature": SUTSpecificationChunk.make("temperature", "t", float),
        "top_p": SUTSpecificationChunk.make("top_p", "p", int),
        "top_k": SUTSpecificationChunk.make("top_k", "k", int),
        "driver": SUTSpecificationChunk.make("driver", "d", str, True),
        "maker": SUTSpecificationChunk.make("maker", "mk", str),
        "provider": SUTSpecificationChunk.make("provider", "pr", str),
        "display_name": SUTSpecificationChunk.make("display_name", "dn", str),
        "reasoning": SUTSpecificationChunk.make("reasoning", "reas", bool),
        "moderated": SUTSpecificationChunk.make("moderated", "mod", bool),
        "driver_code_version": SUTSpecificationChunk.make("driver_code_version", "dv", str),
        "date": SUTSpecificationChunk.make("date", "dt", str),
    }

    pass
