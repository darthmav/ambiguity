---
name: lint
description: Run ruff and mypy checks for the project.
---

# Lint and type-check

Run the project's linting and type-checking tools and report any errors.

```bash
ruff check src/ tests/
mypy src/langgraph_agent/
```

If ruff reports auto-fixable issues, run:

```bash
ruff check --fix src/ tests/
```

Do not change behavior just to silence a warning; use a targeted `# noqa` or `# type: ignore` with a comment when the warning comes from upstream library API drift.
