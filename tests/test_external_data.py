import pytest
from collections import namedtuple

from tenacity import stop_after_attempt

from modelgauge.external_data import GDriveData, LocalData, WebData
from unittest.mock import ANY

from tenacity import wait_none


GDriveFileToDownload = namedtuple("GDriveFileToDownload", ("id", "path"))


def test_web_data_download(mocker):
    mock_download = mocker.patch("urllib.request.urlretrieve")
    web_data = WebData(source_url="http://example.com")
    web_data.download("test.tgz")
    mock_download.assert_called_once_with(
        "http://example.com", "test.tgz", reporthook=ANY
    )


def test_gdrive_data_download(mocker):
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[GDriveFileToDownload("file_id", "file.csv")],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(
        data_source="http://example_drive.com", file_path="file.csv"
    )
    gdrive_data.download.retry.wait = wait_none()
    gdrive_data.download("test.tgz")
    mock_download_folder.assert_called_once_with(
        url="http://example_drive.com", skip_download=True, quiet=ANY, output=ANY
    )
    mock_download_file.assert_called_once_with(id="file_id", output="test.tgz")


def test_gdrive_correct_file_download(mocker):
    """Checks that correct file is downloaded if multiple files exist in the folder."""
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[
            GDriveFileToDownload("file_id1", "different_file.csv"),
            GDriveFileToDownload("file_id2", "file.txt"),
            GDriveFileToDownload("file_id3", "file.csv"),
        ],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(
        data_source="http://example_drive.com", file_path="file.csv"
    )
    gdrive_data.download.retry.wait = wait_none()
    gdrive_data.download("test.tgz")
    mock_download_folder.assert_called_once_with(
        url="http://example_drive.com", skip_download=True, quiet=ANY, output=ANY
    )
    mock_download_file.assert_called_once_with(id="file_id3", output="test.tgz")


def test_gdrive_download_file_with_relative_path(mocker):
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[
            GDriveFileToDownload("file_id", "file.csv"),
            GDriveFileToDownload("nested_file_id", "sub_folder/file.csv"),
        ],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(
        data_source="http://example_drive.com", file_path="sub_folder/file.csv"
    )
    gdrive_data.download.retry.wait = wait_none()
    gdrive_data.download("test.tgz")
    mock_download_file.assert_called_once_with(id="nested_file_id", output="test.tgz")


def test_gdrive_nonexistent_filename(mocker):
    """Throws exception when the folder does not contain any files with the desired filename."""
    mock_download_folder = mocker.patch(
        "gdown.download_folder",
        return_value=[
            GDriveFileToDownload("file_id1", "different_file.csv"),
            GDriveFileToDownload("file_id2", "file.txt"),
        ],
    )
    mock_download_file = mocker.patch("gdown.download")
    gdrive_data = GDriveData(
        data_source="http://example_drive.com", file_path="file.csv"
    )
    gdrive_data.download.retry.wait = wait_none()
    with pytest.raises(RuntimeError, match="Cannot find file"):
        gdrive_data.download("test.tgz")
    mock_download_file.assert_not_called()


def test_local_data_download(mocker):
    mock_copy = mocker.patch("shutil.copy")
    local_data = LocalData(path="origin_test.tgz")
    local_data.download("destintation_test.tgz")
    mock_copy.assert_called_once_with("origin_test.tgz", "destintation_test.tgz")
