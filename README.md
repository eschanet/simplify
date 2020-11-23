# simplify

[![CI status](https://github.com/eschanet/simplify/workflows/CI/badge.svg)](https://github.com/eschanet/simplify/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/simplify-hep/badge/?version=latest)](https://simplify-hep.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/eschanet/simplify/branch/master/graph/badge.svg)](https://codecov.io/gh/eschanet/simplify)
[![PyPI version](https://badge.fury.io/py/simplify-hep.svg)](https://badge.fury.io/py/simplify-hep)
[![python version](https://img.shields.io/pypi/pyversions/simplify-hep.svg)](https://pypi.org/project/simplify-hep/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A package that creates simplified likelihoods from full likelihoods. Currently, only one form of simplified likelihoods is implemented, but the idea is to implement additional versions of the simplified likelihoods, such that the user can chose the one he likes the most (or needs the most).

## Installation

Follow good practice and start by creating a virtual environment

```sh
python3 -m venv simplify
```

and then activating it

```sh
source simplify/bin/activate
```

### Default install

Install the package with pip

```sh
python3 -m pip install simplify-hep[contrib]
```

### Development install

If you want to contribute to `simplify`, install the development version of the package. Fork the repository, clone the fork, and then install

```sh
python3 -m pip install --ignore-installed -U -e .[complete]
```

Next, setup the git pre-commit hook for Black

```sh
pre-commit install
```

You caan run all the tests with

```sh
python3 -m pytest
```

## How to run

### CLI

Run with

```sh
simplify convert -i <fullLH.json> -o <simplifiedLH.json>
```

where `fullLH.json` is the full likelihood you want to convert into a simplified likelihood.

### In python script

You can also use `simplify` in a python script, e.g. to create some validation and cross-check plots and tables.

```py
import pyhf
import json

import simplify

pyhf.set_backend(pyhf.tensorlib, "minuit")
spec = json.load(open("likelihood.json", "r"))

ws = pyhf.Workspace(spec) # ws from full LH

# get model and data for each ws we just created
model = ws.model(modifier_settings = {"normsys": {"interpcode": "code4"},"histosys": {"interpcode": "code4p"},})
data = ws.data(model)

# run fit
fit_result = simplify.fitter.fit(ws)

plt = simplify.plot.pulls(
    fit_result,
    "plots/"
)

plt = simplify.plot.correlation_matrix(
    fit_result,
    "plots/",
    pruning_threshold=0.1
)

tables = simplify.plot.yieldsTable(
    ws,
    "plots/",
    fit_result,
)
```

## Dependencies

Naturally relies heavily on `pyhf`. Part of the code for plotting and validating results is inspired by Alexander Held's [`simplify`](https://github.com/eschanet/simplify/blob/master/src/simplify/fit.py).
