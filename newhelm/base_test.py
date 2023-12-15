from abc import ABC, abstractmethod


class BaseTest(ABC):
    """This is the placeholder base class for all tests."""

    @abstractmethod
    def run(self):
        pass
