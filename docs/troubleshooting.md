# Troubleshooting

> Scaffold stub. Known failures and fixes are added as stages are implemented.

## Common checks

- Validate config: `python -m src.config --show`
- Syntax check: `python -m compileall -q src`
- Fast tests: `python -m pytest tests/test_config.py -q`

## Sections to fill in

- Environment and dependency issues.
- Dataset generation and alignment problems.
- Training instability or checkpoint issues.
- Evaluation/inference artifact problems.
- Colab runtime notes.
