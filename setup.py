from setuptools import setup

extras_require = {"contrib": ["matplotlib"]}
extras_require["test"] = sorted(
    set(
        extras_require["contrib"]
        + [
            "pytest",
            "pytest-cov>=2.6.1",
            "flake8",
            "flake8-bugbear",
            "flake8-import-order",
            "flake8-print",
            "mypy",
            "types-PyYAML",
            "types-tabulate",
            "typeguard~=2.10.0",
            "pydocstyle",
            "black",
        ]
    )
)

extras_require["develop"] = sorted(
    set(extras_require["test"] + ["bump2version", "pre-commit", "twine"])
)

extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))

setup(
    extras_require=extras_require,
    use_scm_version=lambda: {'local_scheme': lambda version: ''},
)
