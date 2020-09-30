# %%
import pyhf
import json
import numpy as np

import simplify

# %%
# for uncertainties on fit results, I think we still need minuit as optimiser (performance isn't a problem for this tool)
pyhf.set_backend(pyhf.tensorlib, "minuit")

# %%
# Get the workspace from model spec

spec = json.load(
    open(
        "/project/ma/packages/simplify/test/EwkOneLeptonTwoBjets2018/BkgOnly.json", "r"
    )
)

# %%

model, data = simplify.model_utils.model_and_data(spec)

# %%
fit_result = simplify.fit.fit((model,data))

# %%
fit_result = fit_result.sort()
# %%
# Correlation matrix
plt = simplify.plot.correlation_matrix(fit_result,"/project/ma/packages/simplify/test/figures/",pruning_threshold=0.2)

# %%
# Pull plot
plt = simplify.plot.pulls(fit_result,"/project/ma/packages/simplify/test/figures/")

print(fit_result)


data = workspace.data(model)
init = model.config.suggested_init()
bounds = model.config.suggested_bounds()
asimov = model.expected_data(pyhf.tensorlib.astensor(init))

# %%
result, minuit = pyhf.infer.mle.fit(
    workspace.data(model),
    model,
    # return_uncertainties=True,
    return_result_obj=True,
)

# %%
print(minuit)
bestfit = result.T
correlations = result_obj.corr

# %%
vals = pyhf.tensorlib.concatenate(
    [
        bestfit[model.config.par_slice(k)]
        for k in model.config.par_order
        # if model.config.param_set(k).constrained
    ]
)

# %%
labels = np.asarray(
    [
        "{}[{}]".format(k, i) if model.config.param_set(k).n_parameters > 1 else k
        for k in model.config.par_order
        # if model.config.param_set(k).constrained
        for i in range(model.config.param_set(k).n_parameters)
    ]
)
# %%
_order = np.argsort(labels)
vals = bestfit[_order]
errors = errors[_order]
labels = labels[_order]
# %%
for label, val in zip(labels, vals):
    print(f"    {label}: {val}")

# %%
for k in model.config.par_order:
    print(f"    {k}: {bestfit[model.config.par_slice(k)]}")


# expected_data = model.expected_data()
# print(expected_data)
