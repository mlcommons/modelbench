from enum import Enum

from pydantic import BaseModel


class SafeTestPersonas(str, Enum):
    NORMAL = "normal"
    SKILLED = "skilled"
    UNSKILLED = "unskilled"


class PersonaResult(BaseModel):
    frac_safe: float
    num_items: int
