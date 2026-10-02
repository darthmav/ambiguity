---
name: lint
description: Run ruff and mypy checks for the project.
---

# Lint and type-check

Run the checks CI runs and report any errors:

```bash
ruff check src/ tests/ serve.py scripts/ spectral_graph/ example_usage.py ollama_client.py
mypy src/langgraph_agent/ serve.py ollama_client.py spectral_graph/
```

If ruff reports auto-fixable issues, add `--fix` to the same `ruff check`.

Do not change behavior just to silence a warning; use a targeted `# noqa` or `# type: ignore` with a comment when the warning comes from upstream library API drift.
