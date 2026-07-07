# Contributing

Contributions are welcome — bug reports, new case studies, gate implementations,
serving improvements, and documentation all help. The bar is the same for every
change, maintainer or first-timer: it passes the automated review, and every claim
it makes is demonstrated by code that runs.

## Development setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/FrankAsanteVanLaarhoven/mlops-production-pipeline.git
cd mlops-production-pipeline
make install   # all dependencies, including dev + notebook tools
make hooks     # git hooks: lint + code review on commit, unit tests on push
```

## The quality bar (what CI enforces)

Every push and pull request runs three jobs; all must pass:

| Check | Command | What it enforces |
|---|---|---|
| Lint | `make lint` | ruff rules (`E,F,W,I,UP,B`), 100-char lines |
| Code review | `make review` | function-level rules R1–R8 (below) |
| Tests | `make test` | unit suite + ≥ 95 % coverage on the core modules |

`tools/code_review.py` is a deterministic AST auditor that inspects **every function
in `src/` and `tools/`** on every run. It enforces:

- **R1** public modules, classes, and functions carry a docstring
- **R2** every parameter and return is type-annotated
- **R3** cyclomatic complexity ≤ 10
- **R4** function length ≤ 60 lines
- **R5** no bare `except:`
- **R6** no mutable default arguments
- **R7** no work-in-progress markers left in source
- **R8** nesting depth ≤ 4

It also fingerprints each function and diffs against the previous run, so the report
(`artifacts/review/report.json`, uploaded as a CI artifact) shows exactly which
functions a change added, modified, or removed. If a rule blocks you and you believe
the rule is wrong, open an issue about the rule — don't work around it.

## Working on the code

- Core logic lives in `src/mlops_pipeline/` and is framework-free; ZenML steps and
  the Ray Serve deployment are thin adapters. Put new logic in the core with unit
  tests, then adapt it — not the other way around.
- Behaviour changes need a test that fails without the change.
- Run `make test-all` before opening a PR if you touched the pipeline or steps —
  it includes the end-to-end ZenML run.
- Local quirk: if your machine exports a `PYTHONPATH` (ROS, conda, etc.), run tests
  as `PYTHONPATH= uv run pytest` to keep foreign pytest plugins out.

## Working on the case study notebook

`notebooks/adult_income_case_study.ipynb` is a research artifact: committed with
outputs, fully reproducible via `make notebook`. Rules for changing it:

- The notebook may only call functions from the installed `mlops_pipeline` package
  for modelling steps — no duplicated training/validation logic in cells.
- Data must come from a **revision-pinned** source with a recorded checksum.
- Model selection must never touch the final test split.
- Re-execute the whole notebook before committing; the committed outputs must be the
  outputs of the committed code.

New case studies (other Hugging Face datasets, other modalities) are very welcome —
follow the same rules and add a `make` target.

## Pull requests

- Keep PRs focused; one concern per PR.
- Describe *why*, link the issue if one exists, and paste the relevant command
  output (test run, smoke test) rather than asserting it passed.
- Generated artifacts (models, drift reports, registries, datasets) are
  git-ignored; don't force-add them.

## Reporting issues

Use the issue templates. For bugs, include the command, the full error, and your
platform; for gate/threshold discussions, include the evidence (metrics, report
excerpts) the proposal is based on.
