from dataclasses import dataclass, field
from newhelm.general import current_timestamp_millis, get_unique_id


@dataclass(frozen=True, kw_only=True)
class TraceableObject:
    """This is the base class for any object that we want to be traceable."""

    id: str = field(default_factory=get_unique_id)
    """Unique identifier of this object."""

    creation_time_millis: int = field(default_factory=current_timestamp_millis)
    """When this object was created."""
