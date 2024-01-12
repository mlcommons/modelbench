import threading
from typing import Dict, Generic, List, Tuple, TypeVar


_T = TypeVar("_T")


class InstanceRegistry(Generic[_T]):
    """Generic class that lets you store and retrieve a set of instances of a given type."""

    def __init__(self) -> None:
        self._lookup: Dict[str, _T] = {}
        self.lock = threading.Lock()

    def register(self, key: str, value: _T):
        """Add value to the registry, ensuring it has a unique key."""
        with self.lock:
            previous = self._lookup.get(key)
            assert previous is None, f"Registry already contains {key} set to {value}."
            self._lookup[key] = value

    def get(self, key: str) -> _T:
        """Retrieve the value stored for key from the registry, raise exception if missing."""
        with self.lock:
            try:
                return self._lookup[key]
            except KeyError:
                known_keys = list(self._lookup.keys())
                raise KeyError(f"No registration for {key}. Known keys: {known_keys}")

    def items(self) -> List[Tuple[str, _T]]:
        """List all items in the registry."""
        with self.lock:
            return list(self._lookup.items())
