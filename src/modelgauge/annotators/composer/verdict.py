from abc import abstractmethod


class Verdict:
    """DAG outputs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a string name for this output, used for routing and debugging."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.name})"
