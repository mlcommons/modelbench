import inspect
import threading
from dataclasses import dataclass
from modelgauge.dependency_injection import inject_dependencies
from modelgauge.secret_values import MissingSecretValues, RawSecrets
from modelgauge.tracked_object import TrackedObject
from typing import Any, Dict, Generic, List, Sequence, Tuple, Type, TypeVar

_T = TypeVar("_T", bound=TrackedObject)


@dataclass(frozen=True)
class FactoryEntry(Generic[_T]):
    """Container for how to initialize an object."""

    cls: Type[_T]
    uid: str
    args: Tuple[Any]
    kwargs: Dict[str, Any]

    def __post_init__(self):
        param_names = list(inspect.signature(self.cls).parameters.keys())
        if not param_names or param_names[0] != "uid":
            raise AssertionError(
                f"Cannot create factory entry for {self.cls} as its first "
                f"constructor argument must be 'uid'. Arguments: {param_names}."
            )

    def __str__(self):
        """Return a string representation of the entry."""
        return f"{self.cls.__name__}(uid={self.uid}, args={self.args}, kwargs={self.kwargs})"

    def make_instance(self, *, secrets: RawSecrets) -> _T:
        """Construct an instance of this object, with dependency injection."""
        args, kwargs = inject_dependencies(self.args, self.kwargs, secrets=secrets)
        result = self.cls(self.uid, *args, **kwargs)  # type: ignore [call-arg]
        assert hasattr(
            result, "uid"
        ), f"Class {self.cls} must set member variable 'uid'."
        assert (
            result.uid == self.uid
        ), f"Class {self.cls} must set 'uid' to first constructor argument."
        return result

    def get_missing_dependencies(
        self, *, secrets: RawSecrets
    ) -> Sequence[MissingSecretValues]:
        """Find all missing dependencies for this object."""
        # TODO: Handle more kinds of dependency failure.
        try:
            inject_dependencies(self.args, self.kwargs, secrets=secrets)
        except MissingSecretValues as e:
            return [e]
        return []


class InstanceFactory(Generic[_T]):
    """Generic class that lets you store how to create instances of a given type."""

    def __init__(self) -> None:
        self._lookup: Dict[str, FactoryEntry[_T]] = {}
        self.lock = threading.Lock()

    def register(self, cls: Type[_T], uid: str, *args, **kwargs):
        """Add value to the registry, ensuring it has a unique key."""

        with self.lock:
            previous = self._lookup.get(uid)
            assert previous is None, (
                f"Factory already contains {uid} set to "
                f"{previous.cls.__name__}(args={previous.args}, "
                f"kwargs={previous.kwargs})."
            )
            self._lookup[uid] = FactoryEntry[_T](cls, uid, args, kwargs)

    def make_instance(self, uid: str, *, secrets: RawSecrets) -> _T:
        """Create an instance using the  class and arguments passed to register, raise exception if missing."""
        entry = self._get_entry(uid)
        return entry.make_instance(secrets=secrets)

    def get_missing_dependencies(
        self, uid: str, *, secrets: RawSecrets
    ) -> Sequence[MissingSecretValues]:
        """Find all missing dependencies for `uid`."""
        entry = self._get_entry(uid)
        return entry.get_missing_dependencies(secrets=secrets)

    def _get_entry(self, uid: str) -> FactoryEntry:
        with self.lock:
            entry: FactoryEntry
            try:
                entry = self._lookup[uid]
            except KeyError:
                known_uids = list(self._lookup.keys())
                raise KeyError(f"No registration for {uid}. Known uids: {known_uids}")
        return entry

    def items(self) -> List[Tuple[str, FactoryEntry[_T]]]:
        """List all items in the registry."""
        with self.lock:
            return list(self._lookup.items())
