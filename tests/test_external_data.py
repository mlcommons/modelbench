from newhelm.external_data import WebData, LocalData


def test_web_data_download(mocker):
    mock_download = mocker.patch("urllib.request.urlretrieve")
    web_data = WebData(source_url="http://example.com")
    web_data.download("test.tgz")
    mock_download.assert_called_once_with("http://example.com", "test.tgz")


def test_local_data_download(mocker):
    mock_copy = mocker.patch("shutil.copy")
    local_data = LocalData(path="origin_test.tgz")
    local_data.download("destintation_test.tgz")
    mock_copy.assert_called_once_with("origin_test.tgz", "destintation_test.tgz")
