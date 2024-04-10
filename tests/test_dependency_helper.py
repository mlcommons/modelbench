import os
import pytest
import time
from modelgauge.data_packing import DataDecompressor, DataUnpacker
from modelgauge.dependency_helper import (
    DependencyVersionMetadata,
    FromSourceDependencyHelper,
)
from modelgauge.external_data import ExternalData
from modelgauge.general import normalize_filename


class MockExternalData(ExternalData):
    """Fully in memory ExternalData that counts download calls."""

    def __init__(self, text, decompressor=None, unpacker=None):
        super().__init__(decompressor=decompressor, unpacker=unpacker)
        self.download_calls = 0
        self.text = text

    def download(self, location):
        self.download_calls += 1
        with open(location, "w") as f:
            f.write(self.text)


class MockDecompressor(DataDecompressor):
    """Test only decompressor that adds characters to the input file."""

    def __init__(self, extra_text: str):
        self.extra_text = extra_text

    def decompress(self, compressed_location, desired_decompressed_filename):
        with open(compressed_location, "r") as f:
            data = f.read()
        with open(desired_decompressed_filename, "w") as f:
            f.write(data + self.extra_text)


class MockUnpacker(DataUnpacker):
    """Test only unpacker that outputs each character in the input file as a separate file."""

    def unpack(self, packed_location: str, desired_unpacked_location: str):
        with open(packed_location, "r") as f:
            data = f.read()
        for i, c in enumerate(data):
            with open(os.path.join(desired_unpacked_location, f"{i}.txt"), "w") as f:
                f.write(c)


# This is the sha256 of a file containing "data-1".
_DATA_1_HASH = "51bbfa74f8660493f40fd72068f63af436ee13c283ca84c373d9690ff2f1f83c"
# This is the sha256 of a file containing "data-2".
_DATA_2_HASH = "00c2022f72beeabc82c8f02099df7abebe43292bac3f44bf63f5827a8c50255a"


@pytest.mark.parametrize("d1_key", ["d1", "d/1"])
class TestFromSource:
    def test_single_read(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData("data-1"),
            "d2": MockExternalData("data-2"),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={}
        )

        # Get the d1 dependency
        d1_path = helper.get_local_path(d1_key)

        normalized_key = normalize_filename(d1_key)
        assert d1_path.endswith(f"{normalized_key}/{_DATA_1_HASH}")
        assert helper.versions_used() == {d1_key: _DATA_1_HASH}
        assert dependencies[d1_key].download_calls == 1
        assert dependencies["d2"].download_calls == 0

        # Ensure the file contains the expected data.
        with open(d1_path, "r") as f:
            d1_from_file = f.read()
        assert d1_from_file == "data-1"

        # Ensure the .metadata file was written
        with open(d1_path + ".metadata", "r") as f:
            metadata = DependencyVersionMetadata.model_validate_json(f.read())
        assert metadata.version == _DATA_1_HASH
        assert metadata.creation_time_millis > 0

    def test_reads_cached(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData("data-1"),
            "d2": MockExternalData("data-2"),
        }

        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={}
        )
        d1_path = helper.get_local_path(d1_key)
        d1_path_again = helper.get_local_path(d1_key)
        assert d1_path == d1_path_again
        assert dependencies[d1_key].download_calls == 1

    def test_update_all(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData("data-1"),
            "d2": MockExternalData("data-2"),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={}
        )
        versions = helper.update_all_dependencies()
        assert versions == {
            d1_key: _DATA_1_HASH,
            "d2": _DATA_2_HASH,
        }
        assert dependencies[d1_key].download_calls == 1
        assert dependencies["d2"].download_calls == 1
        # Nothing has actually been read.
        assert helper.versions_used() == {}

    def test_required_version_already_exists(self, d1_key, tmpdir):
        # Make the old version
        old_dependencies = {
            d1_key: MockExternalData("data-1"),
        }
        old_helper = FromSourceDependencyHelper(
            tmpdir.strpath, old_dependencies, required_versions={}
        )

        # Get the d1 dependency
        old_d1_path = old_helper.get_local_path(d1_key)

        new_dependencies = {
            d1_key: MockExternalData("updated-data-1"),
        }
        new_helper = FromSourceDependencyHelper(
            tmpdir.strpath, new_dependencies, required_versions={d1_key: _DATA_1_HASH}
        )
        new_d1_path = new_helper.get_local_path(d1_key)
        assert old_d1_path == new_d1_path
        # Ensure it was read from storage.
        assert new_dependencies[d1_key].download_calls == 0

    def test_required_version_is_live(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData("data-1"),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={d1_key: _DATA_1_HASH}
        )

        # Get the d1 dependency
        d1_path = helper.get_local_path(d1_key)
        assert d1_path.endswith(_DATA_1_HASH)
        assert dependencies[d1_key].download_calls == 1

    def test_required_version_unavailable(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData("data-1"),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={d1_key: "not-real"}
        )

        with pytest.raises(
            RuntimeError, match=f"version not-real for dependency {d1_key}"
        ):
            helper.get_local_path(d1_key)

    def test_require_older_version(self, d1_key, tmpdir):
        # First write a version of 'd1' with contents of 'data-1'.
        old_dependencies = {
            d1_key: MockExternalData("data-1"),
        }
        old_helper = FromSourceDependencyHelper(
            tmpdir.strpath, old_dependencies, required_versions={}
        )
        old_d1_path = old_helper.get_local_path(d1_key)
        time.sleep(0.05)  # Ensure timestamp of old is actually older.

        # Now write a newer version of d1
        new_dependencies = {
            d1_key: MockExternalData("updated-data-1"),
        }
        new_helper = FromSourceDependencyHelper(
            tmpdir.strpath, new_dependencies, required_versions={}
        )
        # Force reading the new data.
        new_helper.update_all_dependencies()
        new_d1_path = new_helper.get_local_path(d1_key)
        assert old_d1_path != new_d1_path

        # Finally, set up a helper with a required version.
        required_version_helper = FromSourceDependencyHelper(
            tmpdir.strpath, new_dependencies, required_versions={d1_key: _DATA_1_HASH}
        )
        required_version_d1_path = required_version_helper.get_local_path(d1_key)
        assert new_d1_path != required_version_d1_path
        with open(new_d1_path, "r") as f:
            assert f.read() == "updated-data-1"
        with open(required_version_d1_path, "r") as f:
            assert f.read() == "data-1"

    def test_use_newest_version(self, d1_key, tmpdir):
        # First write a version of 'd1' with contents of 'data-1'.
        old_dependencies = {
            d1_key: MockExternalData("data-1"),
        }
        old_helper = FromSourceDependencyHelper(
            tmpdir.strpath, old_dependencies, required_versions={}
        )
        old_d1_path = old_helper.get_local_path(d1_key)
        time.sleep(0.05)  # Ensure timestamp of old is actually older.

        # Now write a newer version of d1
        new_dependencies = {
            d1_key: MockExternalData("updated-data-1"),
        }
        new_helper = FromSourceDependencyHelper(
            tmpdir.strpath, new_dependencies, required_versions={}
        )
        # Force reading the new data.
        new_helper.update_all_dependencies()
        new_d1_path = new_helper.get_local_path(d1_key)
        assert old_d1_path != new_d1_path

        # Finally, set up a helper with no required version
        latest_version_helper = FromSourceDependencyHelper(
            tmpdir.strpath, new_dependencies, required_versions={}
        )
        latest_version_d1_path = latest_version_helper.get_local_path(d1_key)
        assert old_d1_path != latest_version_d1_path
        with open(old_d1_path, "r") as f:
            assert f.read() == "data-1"
        with open(latest_version_d1_path, "r") as f:
            assert f.read() == "updated-data-1"

    def test_decompresses(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData(
                "data-1", decompressor=MockDecompressor(" - decompressed")
            ),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={}
        )

        # Get the d1 dependency
        d1_path = helper.get_local_path(d1_key)

        normalized_key = normalize_filename(d1_key)
        assert d1_path.endswith(f"{normalized_key}/{_DATA_1_HASH}")

        # Ensure the file contains the expected data.
        with open(d1_path, "r") as f:
            f.read() == "data-1 - decompressed"

    def test_unpacks(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData("data-1", unpacker=MockUnpacker()),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={}
        )

        # Get the d1 dependency
        d1_path = helper.get_local_path(d1_key)

        normalized_key = normalize_filename(d1_key)
        assert d1_path.endswith(f"{normalized_key}/{_DATA_1_HASH}")

        assert sorted(os.listdir(d1_path)) == [
            "0.txt",
            "1.txt",
            "2.txt",
            "3.txt",
            "4.txt",
            "5.txt",
        ]
        # Ensure the file contains the expected data.
        with open(os.path.join(d1_path, "0.txt"), "r") as f:
            first_character_of_d1 = f.read()
        assert first_character_of_d1 == "d"

    def test_decompresses_and_unpacks(self, d1_key, tmpdir):
        dependencies = {
            d1_key: MockExternalData(
                "data-1", decompressor=MockDecompressor("ABC"), unpacker=MockUnpacker()
            ),
        }
        helper = FromSourceDependencyHelper(
            tmpdir.strpath, dependencies, required_versions={}
        )

        # Get the d1 dependency
        d1_path = helper.get_local_path(d1_key)

        normalized_key = normalize_filename(d1_key)
        assert d1_path.endswith(f"{normalized_key}/{_DATA_1_HASH}")

        # Decompressed file has "data-1ABC" in it, so it makes 9 files.
        assert len(os.listdir(d1_path)) == 9
        # Ensure the file contains the expected data.
        with open(os.path.join(d1_path, "8.txt"), "r") as f:
            f.read() == "C"
