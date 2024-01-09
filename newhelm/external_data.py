from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from newhelm.data_packing import DataDecompressor, DataUnpacker

from newhelm.general import shell


@dataclass(frozen=True, kw_only=True)
class ExternalData(ABC):
    """Base class for defining a source of external data."""

    decompressor: Optional[DataDecompressor] = None
    unpacker: Optional[DataUnpacker] = None

    @abstractmethod
    def download(self, location):
        pass


@dataclass(frozen=True, kw_only=True)
class WebData(ExternalData):
    """External data that can be trivially downloaded using wget."""

    source_url: str

    def download(self, location):
        shell(["wget", self.source_url, "-O", location])


@dataclass(frozen=True, kw_only=True)
class LocalData(ExternalData):
    """A file that is already on your local machine.

    WARNING: Only use this in cases where your data is not yet
    publicly available, but will be eventually.
    """

    path: str

    def download(self, location):
        shell(["cp", self.path, location])
