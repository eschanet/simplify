# simplify

[![CI status](https://github.com/eschanet/simplify/workflows/CI/badge.svg)](https://github.com/eschanet/simplify/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/simplify-hep/badge/?version=latest)](https://simplify-hep.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/eschanet/simplify/branch/master/graph/badge.svg)](https://codecov.io/gh/eschanet/simplify)
[![PyPI version](https://badge.fury.io/py/simplify-hep.svg)](https://badge.fury.io/py/simplify-hep)
[![python version](https://img.shields.io/pypi/pyversions/simplify-hep.svg)](https://pypi.org/project/simplify-hep/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A package that creates simplified likelihoods from full likelihoods. Currently, only one form of simplified likelihoods is implemented, but the idea is to implement additional versions of the simplified likelihoods, such that varying degrees of simplification can be supported.

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

You can run all the tests with

```sh
python3 -m pytest
```

## How to run

### CLI

Run with e.g.

```sh
simplify convert < fullLH.json > simplifiedLH.json
```

or e.g.

```sh
curl http://foo/likelihood.json | simplify convert
```

where `fullLH.json` is the full likelihood you want to convert into a simplified likelihood. Simplify is able to read/write from/to stdin/stdout.

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

<!-- ## Real-life example

Using the [ATLAS search for electroweakinos in final states with one lepton](https://link.springer.com/article/10.1140/epjc/s10052-020-8050-3), one can download the full analysis likelihood from [HEPData](https://www.hepdata.net/record/ins1755298)

```
pyhf contrib download https://doi.org/10.17182/hepdata.90607.v3/r3 1Lbb-likelihoods && cd 1Lbb-likelihoods
```

Then, produce a simplified version of the full likelihood

```
simplify convert < BkgOnly.json > simplified_likelihood.json
```

Using this simplified likelihood and the provided signal patchset file, the full analysis contour can be reproduced: -->

## Dependencies

Naturally relies heavily on `pyhf`. Part of the code for plotting and validating results is inspired by Alexander Held's [`cabinetry`](https://github.com/alexander-held/cabinetry/).
