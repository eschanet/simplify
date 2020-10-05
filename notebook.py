# %%
import pyhf
import json
import numpy as np

import simplify

# for uncertainties on fit results, I think we still need minuit as optimiser (performance isn't a problem for this tool)
pyhf.set_backend(pyhf.tensorlib, "minuit")

# Get the workspace from model spec

spec = json.load(
    open(
        "test/EwkOneLeptonTwoBjets2018/BkgOnly.json", "r"
    )
)

# %%

cfg = simplify.configuration.load("test/EwkOneLeptonTwoBjets2018/config.yml")

# %%

model, data = simplify.model_utils.model_and_data(spec)

# %%
fit_result = simplify.fitter.fit((model,data))

# %%
# Correlation matrix
plt = simplify.plot.correlation_matrix(fit_result,"test/figures/",pruning_threshold=0.1)

# %%
# Pull plot
plt = simplify.plot.pulls(fit_result,"test/figures/")

# %%
stdevs = simplify.model_utils.calculate_stdev(model,fit_result.bestfit,fit_result.uncertainty,fit_result.corr_mat)

# %%
# plt = siplify.plot.data_MC(cfg,"test/figures/",spec,fit_result)

# %%
yields = simplify.validation.get_yields(cfg, spec, fit_result)

yields

# %%
newspec = simplify.simplified.get_simplified_spec(spec,yields, allowed_modifiers=["lumi"], prune_channels=[])


newspec
