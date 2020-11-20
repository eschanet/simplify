# Contributing to simplify

Your contributions to `simplify` are very much welcome! Thanks a ton for helping out!

## Issues

[Issues](https://github.com/eschanet/simplify/issues) are a good place to report bugs, ask questions, request features, or discuss potential changes to `simplify`.
Before opening a new issue, please have a look through existing issues to avoid duplications.

## Pull requests

It can be helpful to first get into contact via issues before getting started with a [pull request](https://github.com/eschanet/simplify/pulls).
All pull requests are squashed and merged, so feel free to commit as many times as you want to the branch you are working on.
The final commit message should follow the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

## Development environment

For development, install `simplify` with the `[develop]` setup extras.
Then install `pre-commit`

```bash
pre-commit install
```

which will run checks before committing any changes.
You can run all tests for `simplify` with

```bash
python -m pytest
```

All tests are required to pass before any changes can be merged.
