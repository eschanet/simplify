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
                "name": "helm",
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
                        "data": [11],
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
            {"config": {"parameters": [], "poi": "mu_Sig"}, "name": "helm"}
        ],
        "observations": [
            {"data": [42, 9], "name": "region_1"},
            {"data": [10], "name": "region_2"},
        ],
        "version": "1.0.0",
    }
    return spec
