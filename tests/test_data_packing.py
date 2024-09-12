import os
import pytest
from modelgauge.data_packing import (
    GzipDecompressor,
    TarPacker,
    ZipPacker,
    ZstdDecompressor,
)
from tests.utilities import parent_directory


@pytest.mark.parametrize(
    "decompressor,input_filename",
    [
        (GzipDecompressor(), "f1.txt.gz"),
        (ZstdDecompressor(), "f1.txt.zst"),
    ],
)
def test_data_decompression(decompressor, input_filename, parent_directory, tmpdir):
    source_filename = str(parent_directory.joinpath("data", input_filename))
    destination_file = str(os.path.join(tmpdir, "f1.txt"))
    decompressor.decompress(source_filename, destination_file)

    with open(destination_file, "r") as f:
        assert f.read() == "first file.\n"


@pytest.mark.parametrize(
    "unpacker,input_filename",
    [
        (TarPacker(), "two_files.tar.gz"),
        (ZipPacker(), "two_files.zip"),
    ],
)
def test_data_unpacking(unpacker, input_filename, parent_directory, tmpdir):
    source_filename = str(parent_directory.joinpath("data", input_filename))
    destination_dir = str(tmpdir)
    unpacker.unpack(source_filename, destination_dir)

    assert sorted(os.listdir(destination_dir)) == ["f1.txt", "f2.txt"]

    # Check file contents.
    with open(os.path.join(destination_dir, "f1.txt"), "r") as f:
        assert f.read() == "first file.\n"
    with open(os.path.join(destination_dir, "f2.txt"), "r") as f:
        assert f.read() == "second file.\n"
