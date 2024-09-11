import glob
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from modelgauge.data_packing import DataUnpacker
from modelgauge.external_data import ExternalData
from modelgauge.general import current_timestamp_millis, hash_file, normalize_filename
from pydantic import BaseModel, Field
from typing import Dict, Mapping, Optional


class DependencyHelper(ABC):
    """Interface for managing the versions of external data dependencies."""

    @abstractmethod
    def get_local_path(self, dependency_key: str) -> str:
        """Return a path to the dependency, downloading as needed."""

    @abstractmethod
    def versions_used(self) -> Mapping[str, str]:
        """Report the version of all dependencies accessed during this run."""

    @abstractmethod
    def update_all_dependencies(self) -> Mapping[str, str]:
        """Ensure the local system has the latest version of all dependencies."""


class DependencyVersionMetadata(BaseModel):
    """Data object we can store along side a dependency version."""

    version: str
    creation_time_millis: int = Field(default_factory=current_timestamp_millis)


class FromSourceDependencyHelper(DependencyHelper):
    """When a dependency isn't available locally, download from the primary source.

    When used, the local directory structure will look like this:
    ```
    data_dir/
    └── dependency_1/
    │   │   version_x.metadata
    │   └── version_x/
    │   │   │   <dependency's data>
    │   │   version_y.metadata
    │   └── version_y/
    │       │   <dependency's data>
    │       ...
    └── dependency_2/
        │   ...
    ```
    """

    def __init__(
        self,
        data_dir: os.PathLike | str,
        dependencies: Mapping[str, ExternalData],
        required_versions: Mapping[str, str],
    ):
        self.data_dir: os.PathLike | str = data_dir
        """Directory path where all dependencies are stored."""
        self.dependencies: Mapping[str, ExternalData] = dependencies
        """External data dependencies and their keys."""
        self.required_versions: Mapping[str, str] = required_versions
        """A mapping of dependency keys to their required version.
           Version requirements are optional. 
           If no dependencies require a specific version, this is an empty mapping (e.g. `{}`)."""
        self.used_dependencies: Dict[str, str] = {}
        """A mapping of dependency keys to the version used during this run."""

    def get_local_path(self, dependency_key: str) -> str:
        """Returns the file path, unless the dependency uses unpacking, in which case this returns the directory path."""
        assert dependency_key in self.dependencies, (
            f"Key {dependency_key} is not one of the known "
            f"dependencies: {list(self.dependencies.keys())}."
        )
        external_data: ExternalData = self.dependencies[dependency_key]

        normalized_key = normalize_filename(dependency_key)
        version: str
        if dependency_key in self.required_versions:
            version = self.required_versions[dependency_key]
            self._ensure_required_version_exists(
                dependency_key, normalized_key, external_data, version
            )
        else:
            version = self._get_latest_version(normalized_key, external_data)
        self.used_dependencies[dependency_key] = version
        return self._get_version_path(normalized_key, version)

    def versions_used(self) -> Mapping[str, str]:
        return self.used_dependencies

    def update_all_dependencies(self):
        latest_versions = {}
        for dependency_key, external_data in self.dependencies.items():
            normalized_key = normalize_filename(dependency_key)
            latest_versions[dependency_key] = self._store_dependency(
                normalized_key, external_data
            )
        return latest_versions

    def _ensure_required_version_exists(
        self,
        dependency_key: str,
        normalized_key: str,
        external_data: ExternalData,
        version: str,
    ) -> None:
        version_path = self._get_version_path(normalized_key, version)
        if os.path.exists(version_path):
            return
        # See if downloading from the source creates that version.
        stored_version = self._store_dependency(normalized_key, external_data)
        if stored_version != version:
            raise RuntimeError(
                f"Could not retrieve version {version} for dependency {dependency_key}. Source currently returns version {stored_version}."
            )

    def _get_latest_version(self, normalized_key, external_data) -> str:
        """Use the latest cached version. If none cached, download from source."""
        version = self._find_latest_cached_version(normalized_key)
        if version is not None:
            return version
        return self._store_dependency(normalized_key, external_data)

    def _get_version_path(self, normalized_key: str, version: str) -> str:
        return os.path.join(self.data_dir, normalized_key, version)

    def _find_latest_cached_version(self, normalized_key: str) -> Optional[str]:
        metadata_files = glob.glob(
            os.path.join(self.data_dir, normalized_key, "*.metadata")
        )
        version_creation: Dict[str, int] = {}
        for filename in metadata_files:
            with open(filename, "r") as f:
                metadata = DependencyVersionMetadata.model_validate_json(f.read())
            version_creation[metadata.version] = metadata.creation_time_millis
        if not version_creation:
            return None
        # Returns the key with the max value
        return max(
            version_creation.keys(), key=lambda dict_key: version_creation[dict_key]
        )

    def _store_dependency(self, normalized_key, external_data: ExternalData) -> str:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_location = os.path.join(tmpdirname, normalized_key)
            external_data.download(tmp_location)
            version = hash_file(tmp_location)
            final_path = self._get_version_path(normalized_key, version)
            if os.path.exists(final_path):
                # TODO Allow for overwriting
                return version
            if external_data.decompressor:
                decompressed = os.path.join(
                    tmpdirname, f"{normalized_key}_decompressed"
                )
                external_data.decompressor.decompress(tmp_location, decompressed)
                tmp_location = decompressed
            os.makedirs(os.path.join(self.data_dir, normalized_key), exist_ok=True)
            if not external_data.unpacker:
                shutil.move(tmp_location, final_path)
            else:
                self._unpack_dependency(
                    external_data.unpacker, tmp_location, final_path
                )
            metadata_file = final_path + ".metadata"
            with open(metadata_file, "w") as f:
                f.write(DependencyVersionMetadata(version=version).model_dump_json())
            return version

    def _unpack_dependency(
        self, unpacker: DataUnpacker, tmp_location: str, final_path: str
    ):
        os.makedirs(final_path)
        # TODO Consider if we need to handle the single-file case from HELM.
        unpacker.unpack(tmp_location, final_path)
