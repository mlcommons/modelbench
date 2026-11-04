# modelbench Review Guide

This guide is distilled from past pull request reviews by the core team. Review the way they do. Focus on
design, domain clarity, correctness, and tests. Leave formatting to `black`.

## Review Priorities

### CRITICAL

* **Never leak restricted data.** Official, heldback, or practice prompts and their annotations must never
  reach an output, an export, a public artifact, or CI unless they are explicitly marked as allowed. A leak
  costs a lot of money to fix (new prompts have to be written). Flag any code path that could write
  non-exportable prompts or annotations to a file, a log, or a user. Flag tokens for private prompt sets added
  to public CI.
  *e.g. "modelrunner should never put anything in the file that isn't explicitly allowed."*
* **Don't break file formats silently.** Journals, benchmark records, and standards files are read by other
  tools (modellab, modelrunner, the consistency checker) and compared across runs. Any change that isn't
  strictly additive needs a format version number, and readers must handle old and new versions. Renaming
  journal message labels can break the consistency checker.
* **A benchmark's definition and its UID/version change together.** If a change alters what a benchmark,
  evaluator, SUT provider, or prompt set means, its UID or version must change too. Results that share a UID
  must come from the same thing.
  *e.g. "Since we've run a benchmark with the old setup, I don't think we should change the SUT provider
  without changing the UID."*
* **Calibration must cover the full dataset.** Defaults must never let someone recalibrate standards from a
  fraction of the prompts.
* **Don't change released benchmarks.** Scoring, standards, or behavior of a benchmark version that has
  already shipped must not change; create a new version instead.
  *e.g. "wait, should we be making changes to safety 1.2 if its already released?"*
* **Don't count failures as successes.** An item with missing annotations must not be scored as safe or
  silently counted; raise, or record it as a failed item. Question any change that lowers the minimum number
  of required annotations.
* **No confidential material in the public repo.** Flag internal contracts, private prompt data, or other
  non-public details added to docs or code.

### HIGH

* **Domain-driven design.** Objects should match how people talk about the domain. Flag:
  * objects that can exist in an invalid or half-built state ("objects should be constructed in one go");
    data that isn't loaded in `__init__`; `validate()` calls the user has to remember to make
  * mutable UIDs or identity objects. Prefer immutability; use a builder if construction is incremental
  * `if self.is_x: ...` branching that means one class is really two (suggest a subclass)
  * "mystery meat": `dict[str, Any]` metadata, "side information", fields whose only explanation is a comment
  * anemic class hierarchies of static methods where stateful instances would be clearer; "factories" that
    don't return the thing they claim to make
  * law-of-Demeter violations; one module calling another module's private (`_foo`) functions
* **Simplicity / YAGNI.** Push back on speculative generality, registries, global variables, metaprogramming,
  run-time class patching, and new frameworks that duplicate existing ones. Ask what simpler alternatives were
  considered. Build for needs we know we have, except for low-probability, high-cost risks. Praise PRs that
  delete code.
  *e.g. "My general approach to possible futures is to deal with them as they come."*
* **Tests must show intent.** Ask for tests that prove the PR does what it claims, including edge cases and
  backward compatibility (e.g. older journals). Flag:
  * tests that just restate config or implementation ("asserting that the config is the config")
  * tests that depend on hidden coincidences between fixtures; tests coupled to internals instead of the
    object's public behavior
  * tests that mutate globals or `os.environ` without cleanup (suggest `monkeypatch`, fixtures, or try/finally)
  * duplicated test data that must be kept in sync with production data (read it from the source instead)
  * failing tests disabled in production code. Disable them in the tests (e.g.
    `SUTS_THAT_WE_DONT_CARE_ABOUT_FAILING`) and file a card
  * slow tests (long real sleeps or retry waits). Make timing parameters injectable
* **Errors must be actionable.** Don't swallow exceptions. Use `logger.exception(...)` or `raise ... from e`.
  Use specific exception classes, and test for them rather than generic `Exception`. CLI tools should not dump
  stack traces for reasonable user input; catch the error at the top and print a clear message. Different
  user mistakes should produce different messages. Failure reports should summarize (counts, percentages,
  which component) rather than dump long ID lists. Fail before an expensive run, not after it.
* **Robustness in external calls.** Production SUTs/annotators need retries for transient errors and rate
  limits (429, 5xx), with capped exponential backoff. Be skeptical of new timeouts; ask whether they will cause
  needless job failures and whether we'll be able to tell when they fire. HTTP-style endpoints should be
  idempotent.
* **No circular or inverted dependencies.** `modelgauge` must not import from `modelbench`.
* **Fail loudly on bad input.** Unknown keys, prompt sets, or UIDs should raise a clear error, not return
  `""`, `None`, or a silent default that surfaces as a confusing bug later. Registering a duplicate UID should
  error. Use assertions to guard hidden assumptions (e.g. "exactly one completion per response").
* **Encapsulation.** Callers shouldn't need to know internal classifications (e.g. whether a SUT is dynamic or
  pre-registered); the factory/registry decides. Code that uses SUTs should treat them uniformly.

### MEDIUM

* **Naming.** Names should say exactly what something does. Flag:
  * functions whose behavior has drifted from their name
  * `to_x` methods that don't return a whole `x`
  * names that shadow an enclosing function or instance attribute
  * "path" options that are really files (`--output-file`)
  * double-negative booleans (`disable_progress_bar=False`; prefer `enable_...`)
  * abbreviations that differ from the rest of the codebase (`temp` vs `temperature`)
  * the same concept named differently in different places ("heldback" vs "practice", "vendor" vs "maker")
* **Magic values.** Pull thresholds, defaults, and repeated literals (`0.03`, `"</think>"`, `"./run"`) into
  named constants or parameters. Flag magic indices into lists (e.g. `pipeline_segments[2]`).
* **Logging, not print.** Use `logger = logging.getLogger(__name__)` per module. Configure logging once at the
  top of the CLI. Use `logger.info`/`warning` rather than `print`, `warnings.warn`, or `click.echo` for
  diagnostics in long-running or batch code. Remove leftover debug code.
* **Comments.** Comments are a last resort; a concise class docstring explaining what a self-contained
  check or component verifies is fine. Prefer clear names, smaller extracted methods, and tests as
  executable documentation. Flag:
  * comments that duplicate the code or are already out of date
  * historical notes ("this used to...") that belong in git history or the PR description
  * TODOs that won't be done soon; phrase them as notes or file a card
  * a comment inside a long method that marks a method waiting to be extracted
* **Configuration vs secrets.** Ask whether a hardcoded host, region, or URL should be configurable. Don't
  introduce ad-hoc environment variables for configuration; the secrets mechanism reports all missing values at
  startup. At the same time, flag non-secret configuration creeping into `secrets.toml`. SUT definitions should
  be reproducible from version control, not from local files.
* **Return structures.** Prefer named/structured return values over positional tuples of unrelated types.
  Prefer existing domain objects (e.g. `HazardScore`) over ad-hoc dicts. When a nested dict is unavoidable,
  use descriptive variable names (not `k1`, `k2`) and make the key structure clear.
* **Remove cruft.** Ask for dead code, unused classes, unused files (old standards files, unused personas),
  and unnecessary `try`/`except` blocks to be deleted, including matching docs. Ask why a test was deleted.
* **Abstract interfaces.** Use `ABC` with `@abstractmethod` rather than hand-raising `NotImplementedError`, and
  don't give a base-class method a default that hides a missing override (e.g. always returning `True`). Put
  only methods that form a real contract for all subclasses in the base class. Flag required class attributes
  (like `VERSION`) that subclasses could easily forget; enforce them.
* **User-facing docs.** Check that docs match the code: working links, accurate commands, and context at the
  top explaining what the doc is for. Flag inconsistent terminology between docs and code.
* **Don't overwrite outputs.** Output files (results, annotations, journals) should not be silently
  overwritten by the next run; consider timestamps in file names or an `--output-file` option.
* **Pydantic.** Ask whether it's needed. Use it for real JSON I/O; prefer dataclasses elsewhere.
* **Dependencies and CI.** Pin tool versions (e.g. `uv`). Prefer package-manager installs over piping remote
  shell scripts.
* **Scope.** Changes should be whole and focused. Flag autoformatter churn on lines the author didn't edit,
  and unrelated changes mixed into a PR. Suggest follow-up cards for out-of-scope cleanup instead of blocking.

### LOW

* Early returns instead of deeply nested branches.
* Keyword arguments for non-obvious positional arguments.
* Import groups: standard library, third-party, then project code.
* Use `isinstance` asserts and `defaultdict` where they simplify code.
* Type hints, especially return types, on new public methods in `modelgauge`. Return types should name the
  concrete request/response class.
* Pydantic idioms: `model_dump()` (with `exclude_none=True` where appropriate) instead of hand-built dicts.
* File layout: follow the existing convention (e.g. base node types in `nodes.py`, subclasses in their own
  modules). Put test-only helpers in test code or shared `conftest.py`, not in production modules.
* Spell product names consistently (ModelBench vs modelbench).
* Field ordering in definitions: core fields first in a fixed order, then optional groups.
* UID formats: segments separated by dashes; lowercase locale strings (`en_us`).

## Project-Specific Rules

* **SUTs** ("systems under test") are created through the dynamic SUT factories. Factories should hold state
  (secrets, clients) and return real SUT instances. Don't expand the old `SUTS.register` registry. Rich SUT
  UIDs must be canonicalized (`SUTDefinition.canonicalize`) so that equivalent inputs produce one UID; the
  factory should do this, not every caller. SUT options (temperature, max tokens) are not part of the SUT UID.
* **UIDs** go through the `HasUid` machinery, not hand-built string formatting. A UID is immutable. SUT UIDs
  include the explicit model version (`mistral-large-2411`) and the provider. Avoid ambiguous segments: the
  same word (e.g. "official") must not mean both a prompt set and an evaluator.
* **Benchmarks** have one class per version (`...V1_2_1`), each with a `VERSION`. Version-specific behavior
  belongs in that version's class, not in `if version == ...` checks. A benchmark's reference SUTs and
  standards belong to its `BenchmarkDefinition`.
* **Locales** are lowercase strings (`en_us`, `fr_fr`). Only published locales appear in user-facing output.
* **Prompt sets:** only approved prompt sets may be used. Tests and CI use demo prompts or local files, never
  practice or official prompts, and never require private tokens. Exactly one smoke test should exercise real
  remote downloads.
* **Journals** are the audit trail for a run. Record what's needed to reproduce and debug it (test init args,
  SUT options, dependency sources and timestamps, the command run). Don't change existing journal entries
  without versioning.
* **Annotators/evaluators:** ensemble and private annotators can take a long time to spin up; don't add
  timeouts that kill slow but healthy jobs. Annotator exportability is per-prompt metadata.
* **`external_scoring/`** holds scoring code we don't own. Don't suggest refactoring it; review only our code's
  integration with it.
* **The CLI** should be forgiving (DWIM): idempotent where it can be, sensible defaults, validated inputs
  (click validators), clear error messages, and the closest matches suggested for typos.
* **Logging levels** come from the CLI's `--debug` flag; modules shouldn't configure logging themselves.

## Review Tone

* Be direct but collegial. Start with what's good, then the concerns. Short, genuine praise is fine ("Love
  to see it", "net negative line count").
* Ask questions when intent is unclear ("Could you say more about...?", "What's the use case?", "What user's
  problem does this solve?"). Don't assume the author is wrong.
* Explain *why*, grounding it in a principle (DDD, YAGNI, idempotency, Postel's law, reproducibility) or a
  concrete failure scenario.
* Suggest a concrete alternative, with a short code snippet when it helps.
* Separate blocking concerns from suggestions. Mark optional items as such ("Small nit:", "Just a
  suggestion, no need to change this now", "Fine for this PR, but at some point...").
* Under deadline pressure, accept pragmatic shortcuts, but ask for a follow-up card that describes the right
  fix.
* One issue per comment. Keep it short.

## Do Not Flag

* Formatting, line length (120), or import sorting that `black` handles.
* Missing docstrings or comments on clear code.
* Code under `src/modelbench/external_scoring/`, except where our code calls it.
* Generated lock files (`uv.lock`, `poetry.lock`).
* Hypothetical future needs. Don't ask for generality nobody has asked for.
* Micro-optimizations without measurements. Ask for numbers before accepting complexity in the name of
  speed.
