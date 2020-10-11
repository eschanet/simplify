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

#%%
original_ws = pyhf.Workspace(spec)
pruned_ws = original_ws.prune(channels=['SRLMEM_mct2','SRMMEM_mct2','SRHMEM_mct2'])
no_staterror_ws = original_ws.prune(modifiers=['staterror_SRLMEM_mct2','staterror_SRMMEM_mct2','staterror_SRHMEM_mct2'])

pruned_model = pruned_ws.model(modifier_settings = {"normsys": {"interpcode": "code4"},"histosys": {"interpcode": "code4p"},},poi_name = None)
pruned_data = pruned_ws.data(pruned_model)
original_model = original_ws.model(modifier_settings = {"normsys": {"interpcode": "code4"},"histosys": {"interpcode": "code4p"},},poi_name = None)
original_data = original_ws.data(original_model)
no_staterror_model = no_staterror_ws.model(modifier_settings = {"normsys": {"interpcode": "code4"},"histosys": {"interpcode": "code4p"},},poi_name = None)
no_staterror_data = no_staterror_ws.data(no_staterror_model)


pruned_fit_result = simplify.fitter.fit((pruned_model,pruned_data))
original_fit_result = simplify.fitter.fit((original_model,original_data))
no_staterror_fit_result = simplify.fitter.fit((no_staterror_model,no_staterror_data))

pruned_fit_result.labels
original_fit_result.labels
no_staterror_fit_result.labels.size
#%%
# model, data = simplify.model_utils.model_and_data(spec)

# fixed_pars = model.config.suggested_fixed()
# inits = model.config.suggested_init()
# zip(inits, fixed_pars)

#%%
yields = simplify.yields.get_yields((no_staterror_model,no_staterror_data), pruned_fit_result)
yields.yields['SRHMEM_mct2']
np.sum(yields.yields['SRMMEM_mct2'], axis=0)
yields.uncertainties['SRMMEM_mct2']

#%%
# Yields table
# simplify.plot.yieldsTable(spec, "test/figures/", pruned_fit_result)

# %%
# Correlation matrix
plt = simplify.plot.correlation_matrix(pruned_fit_result,"test/figures/",pruning_threshold=0.1)

# %%
# Pull plot
plt = simplify.plot.pulls(pruned_fit_result,"test/figures/")

# %%
stdevs = simplify.model_utils.calculate_stdev(model,fit_result.bestfit,fit_result.uncertainty,fit_result.corr_mat)

# %%
plt = siplify.plot.data_MC(cfg,"test/figures/",spec,fit_result)

# %%
# newspec = simplify.simplified.get_simplified_spec(spec,yields, allowed_modifiers=["lumi"], prune_channels=[])
# newspec
