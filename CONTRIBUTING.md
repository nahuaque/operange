# Developing Operange

Use Python 3.10+; the local development default is Python 3.13.

```bash
uv sync --locked --group dev --group docs
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked bandit -q -r src examples tests
uv run --locked --group docs sphinx-build -W -b html docs docs/_build/html
uv build
uv run --locked twine check dist/*
```

`uv sync --locked --extra cvxpy` enables the optional SDP comparisons. The
residopt research remains parked; its notes describe an optional sibling-source
experiment and do not make residopt a dependency of Operange. Missing optional
backends produce explicit skips or unsupported results.

CI also installs the wheel into a clean environment and runs copied consumers
outside this checkout, verifying independence from `updatesupport`, old package
imports and repository examples. It runs on Python 3.10–3.13.

Keep private design material in `notes/private/`, which is ignored by Git and
excluded from distributions. Public design notes belong in `notes/`; supported
user-facing behavior belongs in `docs/`. Examples may show downstream economics
and reporting, but those concerns do not belong in the installed package.

## Publishing

The distribution and import namespace are both `operange`. The manual
`.github/workflows/publish.yml` workflow follows the `updatesupport` release
pattern: choose `testpypi` (the default) or `pypi` when starting **Publish** in
GitHub Actions. Lint, tests (including the optional CVXPY comparisons), security,
documentation and distribution checks run before upload. Pushes and tags do not
publish automatically. PyPI attestations are disabled to match `updatesupport`.

Configure a GitHub Trusted Publisher on each package index you intend to use:

| Field | Value |
| --- | --- |
| Project | `operange` |
| Owner | `nahuaque` |
| Repository | `operange` |
| Workflow filename | `publish.yml` |
| Environment | `testpypi` on TestPyPI; `pypi` on PyPI |

Create the corresponding GitHub environments. For a first release, configure a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/);
for an existing project, add the publisher in its Publishing settings. The
workflow uses OIDC and does not require an API-token secret. TestPyPI and PyPI
publisher registrations are separate.

Set the intended version in `pyproject.toml`, refresh `uv.lock` and commit the
release state before dispatching the workflow. Current initial version: 0.1.0.
