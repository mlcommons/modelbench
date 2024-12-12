import pathlib

import tomli


# TODO: If we plan to keep static content in modelbench, we need to add tests to make sure static content for
#  relevant objects exists.
class StaticContent(dict):
    def __init__(self, path=pathlib.Path(__file__).parent / "templates" / "content"):
        super().__init__()
        self.path = path
        for file in (path).rglob("*.toml"):
            with open(file, "rb") as f:
                try:
                    data = tomli.load(f)
                except tomli.TOMLDecodeError as e:
                    raise ValueError(f"failure reading {file}") from e
                duplicate_keys = set(self.keys()) & set(data.keys())
                if duplicate_keys:
                    raise Exception(f"Duplicate tables found in content files: {duplicate_keys}")
                self.update(data)

    def update_custom_content(self, custom_content_path: pathlib.Path):
        custom_content = StaticContent(custom_content_path)
        for table in custom_content:
            if table not in self:
                raise ValueError(
                    f"Unknown table {table} in custom content from {custom_content_path}; doesn't match {list(self.keys())} from {self.path}"
                )
            self[table].update(custom_content[table])
