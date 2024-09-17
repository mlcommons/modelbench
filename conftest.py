def pytest_addoption(parser):
    parser.addoption(
        "--expensive-tests",
        action="store_true",
        dest="expensive-tests",
        help="enable expensive tests",
    )
