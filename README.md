# simplify

A package that creates simplified likelihoods from full likelihoods.

Relies on `pyhf` and is inspired quite a bit by Alexander Held's [`cabinetry`](https://github.com/alexander-held/cabinetry/blob/master/src/cabinetry/fit.py) for plotting and validation of results.

### How to run

Run with `simplify convert -i fullLH.json -o simplifiedLH.json` where `fullLH.json` is the full likelihood you want to convert into a simplified likelihood.  