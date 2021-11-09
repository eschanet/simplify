import awkward1 as ak
import numpy as np

from simplify import simplified, yields


def test_get_simplified_spec(
    example_full_analysis_likelihood, example_simplified_analysis_likelihood
):

    ylds = yields.Yields(
        regions=['SRLMEM_mct2'],
        samples=[
            'diboson',
            'multiboson',
            'singletop',
            'ttbar',
            'tth',
            'ttv',
            'vh',
            'wjets',
            'zjets',
        ],
        yields={
            'SRLMEM_mct2': np.asarray(
                [
                    [3.81835159e-01, 8.29630721e-02, 1.82488713e-01],
                    [0.00000000e00, 4.39047681e-03, 8.58322892e-05],
                    [1.19221557e01, 7.45065852e00, 4.70797363e00],
                    [1.12848942e00, 1.07669201e00, 7.45587698e-01],
                    [4.14071746e-02, 4.21658868e-02, 3.47387836e-02],
                    [5.44315116e-01, 2.52051808e-01, 2.30325982e-01],
                    [9.73768735e-02, 2.90478572e-01, 4.60765010e-01],
                    [1.42638837e00, 1.82756453e00, 7.67691098e-01],
                    [1.35629924e-01, 6.63743835e-02, 4.54261930e-02],
                ]
            )
        },
        uncertainties={'SRLMEM_mct2': ak.from_iter([23.6, 25.1, 14.6])},
        data={'SRLMEM_mct2': np.asarray([16.0, 11.0, 7.0])},
    )

    spec = simplified.get_simplified_spec(
        example_full_analysis_likelihood, ylds, allowed_modifiers=[], prune_channels=[]
    )

    assert spec == example_simplified_analysis_likelihood
