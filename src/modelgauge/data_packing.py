import os
import tempfile
import zstandard
from abc import ABC, abstractmethod
from modelgauge.general import shell


class DataDecompressor(ABC):
    """Base class for a method which decompresses a single file into a single file."""

    @abstractmethod
    def decompress(self, compressed_location, desired_decompressed_filename: str):
        pass


class GzipDecompressor(DataDecompressor):
    def decompress(self, compressed_location: str, desired_decompressed_filename: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Copy file to a temp directory to not pollute original directory.
            unzipped_path = os.path.join(tmpdirname, "tmp")
            gzip_path = unzipped_path + ".gz"
            shell(["cp", compressed_location, gzip_path])
            # gzip writes its output to a file named the same as the input file, omitting the .gz extension.
            shell(["gzip", "-d", gzip_path])
            shell(["mv", unzipped_path, desired_decompressed_filename])


class ZstdDecompressor(DataDecompressor):
    def decompress(self, compressed_location: str, desired_decompressed_filename: str):
        dctx = zstandard.ZstdDecompressor()
        with open(compressed_location, "rb") as ifh:
            with open(desired_decompressed_filename, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)


class DataUnpacker(ABC):
    """Base class for a method that converts a single file into a directory."""

    @abstractmethod
    def unpack(self, packed_location: str, desired_unpacked_dir: str):
        pass


class TarPacker(DataUnpacker):
    def unpack(self, packed_location: str, desired_unpacked_dir: str):
        shell(["tar", "xf", packed_location, "-C", desired_unpacked_dir])


class ZipPacker(DataUnpacker):
    def unpack(self, packed_location: str, desired_unpacked_dir: str):
        shell(["unzip", packed_location, "-d", desired_unpacked_dir])
