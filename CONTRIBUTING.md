<!--SPDX-License-Identifier: MIT-->
<!--Version: v1.1.0-->

# Contributing

SurroGrid is developed on GitHub: <https://github.com/tum-ens/SurroGrid>. This repository follows the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to take part

- **Use**: no notification is required, but please add yourself to [USERS.cff](USERS.cff).
- **Comment**: report bugs and ideas as GitHub issues; add yourself to [CITATION.cff](CITATION.cff) if you wish.
- **Contribute and maintain**: add code through pull requests and add yourself to [CITATION.cff](CITATION.cff).

## Workflow

1. Open (or pick) a GitHub issue that describes the problem.
2. Branch from `develop` (`main` holds the released version): `feature-<issue>-<short-description>` or
   `hotfix-<issue>-<short-description>`.
3. Commit small logical changes with imperative commit messages that reference the issue (`#42`).
4. Run the tests and the linter (below), update the documentation, and record user-visible changes in
   [CHANGELOG.md](CHANGELOG.md).
5. Open a pull request against `develop` (`Closes #<issue>` in the description) and ask for a review.

## Development setup and checks (GridExpand)

```bash
cd GridExpand
uv sync                              # environment incl. the dev group (pytest, ruff)
uv run pytest -q                     # unit tests; they never connect to a real database
uv run ruff check src tests scripts  # lint (project ruff config)
```

See [GridExpand/README.md](GridExpand/README.md#testing) for the opt-in database tests and the regression harness
(`GridExpand/tests/regression/`), which must only run against a sandbox database. Never run pipeline or
maintenance commands against a production database to try something out; `gridexpand <command> --help` is safe
(arguments are parsed before any connection). GridForecast has no shared test suite yet.

## Code and documentation style

- Python 3.12, PEP 8, the ruff configuration of `GridExpand/pyproject.toml`; vendored code in
  `GridExpand/src/gridexpand/allocation/external/` keeps its upstream style.
- Concise Google-style docstrings for public modules, classes and functions (`Args`, `Returns`, `Raises`).
- Directories only through `gridexpand.paths`; scientific values only in scenario YAMLs (`GridExpand/config/`).
- Documentation lives in `GridExpand/docs/` ([index](GridExpand/docs/README.md)); keep commands, flags and paths
  in it consistent with `--help`. Historical run notes go to `GridExpand/docs/research/<date>_<topic>.md`; AI
  agent plans and handovers stay outside the repository (see [AGENTS.md](AGENTS.md)).
