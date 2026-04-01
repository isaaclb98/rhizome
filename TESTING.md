# Testing

Rhizome uses `pytest` with the following conventions:

- **Test location:** `tests/` directory
- **File naming:** `test_<module>.py`
- **Run tests:** `pytest` or `pytest -v`
- **With coverage:** `pytest --cov=rhizome --cov-report=term-missing`

## Philosophy

100% test coverage is the goal. Tests make vibe coding safe. When writing new functions, write a corresponding test. When fixing a bug, write a regression test. When adding error handling, write a test that triggers the error.

## Test Layers

| Layer | What | Where |
|-------|------|-------|
| Unit | Individual functions, classes | `tests/test_*.py` |
| Integration | Module-to-module flows | `tests/test_*_integration.py` |
| E2E | Full `rhizome ingest` + `rhizome traverse` | Not yet implemented |

## Conventions

- Use `pytest` fixtures for shared setup
- Mock external dependencies (Qdrant, HuggingFace API)
- Test the public interface (not internal implementation details)
- Each test should be independently runnable
