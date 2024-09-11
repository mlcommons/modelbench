# Publishing
We use [Poetry](https://python-poetry.org/) for publishing ModelGauge and its plugins.

## Configuring Poetry

This will add the [poetry-bumpversion](https://github.com/monim67/poetry-bumpversion?tab=readme-ov-file) plugin to your
global Poetry installation.
```shell
poetry self add poetry-bumpversion
```

## Publishing

1. Bump the version of ModelGauge and all plugins by using `poetry version <version>`, where `<version>` is one of:
"patch", "minor", or "major". Note that this will bump the versions of all plugins referenced in pyproject.toml
as well.
1. Commit those version changes, make a PR and merge it into main.
1. Check out the version of main corresponding to your PR. Run `poetry run pytest --expensive-tests` to ensure all tests pass. If they don't, fix the tests and return to the previous step.
1. Tag the commit with the version number you just created, prefixed by `v`, e.g. `git tag v0.2.6`.
1. `git push origin <your tag>`.
1. In Github [create a new release](https://github.com/mlcommons/modelgauge/releases/new).
   1. Select the tag you just created.
   2. Click "Generate release notes".
   3. Edit the automatically generated text to be more human friendly, removing boring things and trimming the text as needed.
   4. For now, also select "Set as a pre-release".
   5. Publish the release.
1. In your local repository use `poetry run python publish_all.py` to automatically build and publish all packages.
   If you're having auth troubles, [get a PyPI token](https://pypi.org/help/#apitoken) and tell poetry to use it with `poetry config pypi-token.pypi YOURTOKEN`

