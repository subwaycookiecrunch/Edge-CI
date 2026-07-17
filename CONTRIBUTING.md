# Contributing

## Running tests

```bash
pip install -e '.[test]'
pytest
```

All tests are offline. No hardware, no model files, no network calls.

## What's in scope

- Bug fixes with a reproducer
- Test coverage for untested paths
- Documentation corrections (typos, dead links, unclear instructions)
- Adapter fixes for new `llama-bench` output schemas

## What's not in scope right now

- New runtime adapters (MLX, Ollama, Core ML)
- Dashboard or web UI
- Hosted runner infrastructure
- AI-generated analysis or diagnosis features

If you want to propose something outside this list, open an issue first.

## Style

- No runtime dependencies beyond `click`, `rich`, and `tomli` (Python < 3.11)
- No NumPy, SciPy, or external stats libraries in the core
- No network calls from the tool itself
- Tests must pass with `pytest` and no hardware present
- Keep docstrings factual. Don't explain what's obvious from the signature.

## Commits

One logical change per commit. Write the subject line as an imperative sentence.
Good: `Fix MTL backend canonicalization for b10052`
Bad: `Updated adapter to handle new backend string`
