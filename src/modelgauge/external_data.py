import requests  # type: ignore
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import gdown  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential

from modelgauge.data_packing import DataDecompressor, DataUnpacker


@dataclass(frozen=True, kw_only=True)
class ExternalData(ABC):
    """Base class for defining a source of external data.

    Subclasses must implement the `download` method."""

    decompressor: Optional[DataDecompressor] = None
    unpacker: Optional[DataUnpacker] = None

    @abstractmethod
    def download(self, location):
        pass


@dataclass(frozen=True, kw_only=True)
class WebData(ExternalData):
    """External data that can be trivially downloaded using wget."""

    source_url: str
    headers: Optional[Dict] = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1),
        reraise=True,
    )
    def download(self, location):
        if self.headers:
            response = requests.get(self.source_url, headers=self.headers)
        else:
            response = requests.get(self.source_url)
        if response.ok:
            with open(location, "wb") as f:
                f.write(response.content)
        else:
            raise RuntimeError(
                f"failed to fetch {self.source_url} with headers={self.headers}.\nResponse status: {response.status_code}: {response.text}"
            )


@dataclass(frozen=True, kw_only=True)
class GDriveData(ExternalData):
    """File downloaded using a google drive folder url and a file's relative path to the folder."""

    data_source: str
    file_path: str

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=3, min=15),
        reraise=True,
    )
    def download(self, location):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty folder downloaded to tmpdir
            available_files = gdown.download_folder(url=self.data_source, skip_download=True, quiet=True, output=tmpdir)
        # Find file id needed to download the file.
        for file in available_files:
            if file.path == self.file_path:
                gdown.download(id=file.id, output=location)
                return
        raise RuntimeError(f"Cannot find file with name {self.file_path} in google drive folder {self.data_source}")


@dataclass(frozen=True, kw_only=True)
class LocalData(ExternalData):
    """A file that is already on your local machine.

    WARNING: Only use this in cases where your data is not yet
    publicly available, but will be eventually.
    """

    path: str

    def download(self, location):
        shutil.copy(self.path, location)
