# Contributing to Energy-Oracle

Thank you for considering a contribution!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Energy-Oracle.git
cd Energy-Oracle
pip install -r requirements.txt
cp .env.example .env
make train      # train initial model
make run        # start dev server at http://localhost:8000
```

## Running Tests

```bash
make test
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
make lint
```

All PRs must pass `ruff check .` with zero errors before merging.

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Add or update tests for any changed behaviour.
3. Run `make lint` and `make test` locally before opening a PR.
4. Write a clear PR description explaining what and why.

## Commit Convention

```
feat(scope): short description
fix(scope): short description
test(scope): short description
docs(scope): short description
chore(scope): short description
```

## Reporting Issues

Open a GitHub Issue with:
- Expected vs actual behaviour
- Steps to reproduce
- Python and OS version
