---
name: test
description: Run the full pytest suite for the 4-agent system.
---

# Run tests

Run the test suite with the stub LLM so no API keys or Ollama instance is needed.

```bash
python -m pytest tests/ -v
```

After running, report the pass/fail count and any warnings worth noting.
