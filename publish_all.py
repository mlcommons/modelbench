import pathlib
import subprocess

all_paths = pathlib.Path(__file__).parent.glob("**/pyproject.toml")

for path in all_paths:
    if ".venv" in str(path):
        continue
    build_command = [
        "poetry",
        "build",
        "--no-interaction",
        "-C",
        str(path.parent.absolute()),
    ]
    publish_command = [
        "poetry",
        "publish",
        "--no-interaction",
        "--skip-existing",
        "-C",
        str(path.parent.absolute()),
    ]

    subprocess.run(build_command, check=True)
    subprocess.run(publish_command, check=True)
