from abc import ABC, abstractmethod

from newhelm.general import shell


class ExternalData(ABC):
    """Base class for defining a source of external data."""

    @abstractmethod
    def download(self, location):
        pass


class WebData(ExternalData):
    """External data that can be trivially downloaded using wget."""

    def __init__(self, source_url):
        self.source_url = source_url

    def download(self, location):
        shell(["wget", self.source_url, "-O", location])
