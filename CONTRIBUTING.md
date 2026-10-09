# Contributing

Contributions to `cnmf-usage` should preserve the package's numerical behavior unless a numerical change is intentional and documented.

## Development setup

Install the locked development environment:

```bash
uv sync --group dev
```

## Tests

Run the public test suite:

```bash
uv run pytest -q tests/
```

## Code quality

Check linting and formatting:

```bash
uvx ruff check src tests
uvx ruff format --check src tests
```

To apply formatting:

```bash
uvx ruff format src tests
```

## Numerical changes

The numerical method is part of the package contract.

Changes that can affect any of the following require numerical regression testing:

- gene matching or gene order
- count-matrix handling
- scaling
- fixed-H NMF fitting
- program-weight calculation
- usage normalization

Maintainers must run the private frozen real-data regression suite before merging a change that can affect numerical output.

Intentional numerical changes must be documented under a `Numerical changes` heading in `CHANGELOG.md`. They should not be released silently as a patch-level change.

## Pull requests

Before submitting or merging a pull request:

1. Run Ruff.
2. Run the public test suite.
3. Add or update tests for changed behavior.
4. Update `CHANGELOG.md` for user-visible changes.
5. Run the private regression suite when numerical output can change.
