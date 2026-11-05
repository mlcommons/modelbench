# scoring-science

Scoring science for the Safety 1.2.1 and Security 1.1 benchmarks, owned and
maintained by the science team. The modelbench repository consumes it as a
library and owns everything around it: the typed boundary, the adapters into
the benchmark runner, and the conformance tests.

**This directory is a prototype**: it simulates the science team's future
repository, wired in as a uv *path* dependency. When their repository exists,
the `[tool.uv.sources]` entry in modelbench's `pyproject.toml` changes from a
path to `{ git = "...", rev = "<hash>" }` and this directory is deleted.
Until then: treat it as external code. Do not refactor it from the
modelbench side; change it only as the science team would (a refit, a scoring
change), and always bump the version below.

## The contract

1. **Plain data only.** The library imports nothing from modelbench or
   modelgauge, reads no run state, and touches no network or filesystem
   except its own shipped data files. Inputs are JSON-shaped dicts; outputs
   are JSON-serializable dicts.
2. **Entry points.** The engineering side calls exactly these, and nothing else:
   - `scoring_science.safety_v_1_2_1_scoring.score`
   - `scoring_science.security_v_1_1_scoring.score`
   - the `PublicationBlocked` exception each benchmark module raises

   `score` selects, loads and verifies this package's own fitted file from
   the prompt set and locale the caller names, and returns the file's SHA-256
   inside the result under `difficulty_and_discrimination_sha256`. The
   lower-level functions (`score_sut`, `score_sut_pair`, `score_form`) remain
   public for the conformance tests' math checks, but they are not the
   supported entry.
3. **Output shapes are frozen.** Every returned dict must parse against the
   typed models in ``modelbench.scoring_interface.interface`` (Pydantic,
   strict); modelbench runs conformance tests against them before accepting
   any version bump. A release that changes a shape fails those tests, by
   design.
4. **Data files ship here.** The difficulty-and-discrimination JSON files and
   their pinned SHA-256 digests live in this package. A refit replaces a
   file and its digest together with a version bump, in one commit.

The full field-by-field contract for each benchmark is in
`safety_v_1_2_1_scoring.md` and `security_v_1_1_scoring.md` (written before
the split; module paths there are now `scoring_science.*`, and the
runner-side record classes now live in `modelbench.scoring_interface`).

## Versioning

Every change to code or data files bumps `version` in `pyproject.toml`.
Modelbench pins one exact version (path now, git hash later) and upgrades
only through a pull request that runs its conformance suite:

1. Bump `version` here; commit the change (code, data files, digest pins).
2. On the modelbench side: update the pinned source, run `uv lock`, then
   `uv run pytest tests/modelbench_tests/scoring_interface/`.
3. Golden numbers in that suite change only deliberately, with a science-side
   explanation in the upgrade pull request.
