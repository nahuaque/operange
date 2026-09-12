# Installation and development

Operange requires Python 3.10 or newer. NumPy and SciPy are its runtime
dependencies; installing the package installs them automatically.

## Install a release

```bash
python -m pip install operange
```

Then follow the [shared-steam tutorial](index.md). For the changes in each
published version, see the
[release notes](https://github.com/nahuaque/operange/releases).

## Install from a checkout

From the repository root:

```bash
python -m pip install .
```

For editable development with the repository's locked environment:

```bash
uv sync --locked --group dev --group docs
uv run python -m examples.steam_header
```

The [worked examples](examples.md) use the public package API. Their source files
live in the repository and are not installed with the wheel.

## Build a wheel

```bash
uv build --wheel --out-dir dist
```

Install the resulting `.whl` file with `python -m pip install PATH_TO_WHEEL`.
Use the filename produced by the build for the version in your checkout.

## Run the checks

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked bandit -q -r src examples tests
uv run --locked --group docs sphinx-build -W -b html docs docs/_build/html
```

CI also builds the distribution and runs copied consumer examples outside the
checkout, using an isolated Python environment with only the installed wheel
and its declared dependencies. Those checks cover numerical results, evidence,
witness replay and compatibility with saved artifacts on Python 3.10–3.13.

The optional `cvxpy` extra enables [prepared dispatch and convex intersection
support](convex-backends.md), as well as experimental backend comparisons.
Default dispatch and the other built-in primitive geometries use NumPy/SciPy.
