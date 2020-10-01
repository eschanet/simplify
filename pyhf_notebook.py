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

model, data = simplify.model_utils.model_and_data(spec)
fit_result = simplify.fitter.fit((model,data))


# %%

yields_combined = model.main_model.expected_data(
    fit_result.bestfit, return_by_sample=True
)
yields_combined
