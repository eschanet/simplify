import json

import pytest


@pytest.fixture
def example_spec():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [112.429786],
                        "modifiers": [
                            {
                                "data": [8.234677],
                                "name": "staterror_SR",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "mu_Sig",
                                "type": "normfactor",
                            },
                        ],
                        "name": "signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {
                            "name": "staterror_SR",
                            "fixed": True,
                            "inits": [1.1],
                        }
                    ],
                    "poi": "mu_Sig",
                },
                "name": "single bin",
            }
        ],
        "observations": [{"data": [691], "name": "SR"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_multibin():
    spec = {
        "channels": [
            {
                "name": "region_1",
                "samples": [
                    {
                        "data": [40, 8],
                        "modifiers": [
                            {
                                "data": [7, 3],
                                "name": "staterror_region_1",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "mu_Sig",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            },
            {
                "name": "region_2",
                "samples": [
                    {
                        "data": [10],
                        "modifiers": [
                            {
                                "data": [2],
                                "name": "staterror_region_2",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "mu_Sig",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            },
        ],
        "measurements": [
            {"config": {"parameters": [], "poi": "mu_Sig"}, "name": "multi bin"}
        ],
        "observations": [
            {"data": [42, 9], "name": "region_1"},
            {"data": [10], "name": "region_2"},
        ],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_shapefactor():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [30, 12],
                        "modifiers": [
                            {
                                "data": None,
                                "name": "shape factor",
                                "type": "shapefactor",
                            },
                            {
                                "data": None,
                                "name": "mu_Sig",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {"parameters": [], "poi": "mu_Sig"},
                "name": "shapefactor fit",
            }
        ],
        "observations": [{"data": [32, 11], "name": "SR"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_with_background():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [43],
                        "modifiers": [
                            {
                                "data": [5],
                                "name": "staterror_SR",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "mu_Sig",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    },
                    {
                        "data": [146],
                        "modifiers": [
                            {
                                "data": [7],
                                "name": "staterror_SR",
                                "type": "staterror",
                            }
                        ],
                        "name": "Bkg",
                    },
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {
                            "name": "mu_Sig",
                            "bounds": [[0, 10]],
                            "inits": [1.0],
                        }
                    ],
                    "poi": "mu_Sig",
                },
                "name": "signal+background",
            }
        ],
        "observations": [{"data": [150], "name": "SR"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_full_analysis_likelihood():
    return json.load(open('tests/helpers/reference/BkgOnly.json'))


@pytest.fixture
def example_simplified_analysis_likelihood():
    return json.load(open('tests/helpers/reference/simplified_BkgOnly.json'))
