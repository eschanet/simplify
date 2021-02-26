import copy
import logging
from typing import Any, Dict, List

import awkward1 as ak
import pyhf

from . import yields


log = logging.getLogger(__name__)


def get_simplified_spec(
    spec: Dict[str, Any],
    ylds: yields.Yields,
    allowed_modifiers: List[str],
    prune_channels: List[str],
    include_signal: bool = False,
) -> pyhf.workspace:

    newspec = {
        'channels': [
            {
                'name': channel['name'],
                'samples': [
                    {
                        'name': 'Bkg',
                        'data': ylds.yields[channel['name']]
                        .sum(axis=0)
                        .flatten()
                        .tolist(),
                        "modifiers": [
                            {
                                "data": {
                                    "hi_data": (
                                        ylds.yields[channel['name']].sum(axis=0)
                                        + ak.to_numpy(
                                            ylds.uncertainties[channel['name']]
                                        )
                                    )
                                    .flatten()
                                    .tolist(),
                                    "lo_data": (
                                        ylds.yields[channel['name']].sum(axis=0)
                                        - ak.to_numpy(
                                            ylds.uncertainties[channel['name']]
                                        )
                                    )
                                    .flatten()
                                    .tolist(),
                                },
                                "name": "totalError",
                                "type": "histosys",
                            }
                        ],
                    }
                ],
            }
            for channel in spec['channels']
            if channel['name'] not in prune_channels
        ],
        'measurements': [
            {
                'name': measurement['name'],
                'config': {
                    'parameters': [
                        {
                            "auxdata": [1.0],
                            "bounds": [[0.915, 1.085]],
                            "fixed": True,  # this is the important part
                            "inits": [1.0],
                            "name": "lumi",
                            "sigmas": [0.017],
                        }
                    ]
                    + [
                        dict(
                            parameter,
                            name=parameter['name'],
                        )
                        for parameter in measurement['config']['parameters']
                        if parameter['name'] in allowed_modifiers
                    ],
                    'poi': 'mu_Sig',
                },
            }
            for measurement in spec['measurements']
        ],
        'observations': [
            dict(
                copy.deepcopy(observation),
                name=observation['name'],
            )
            for observation in spec['observations']
        ],
        'version': spec['version'],
    }

    if include_signal:
        channels_with_signal = [
            {
                'name': c['name'],
                'samples': c['samples']
                + [
                    {
                        "name": "Signal",
                        "data": [0]
                        * len(ylds.yields[c['name']].sum(axis=0).flatten().tolist()),
                        "modifiers": [
                            {"data": None, "name": "mu_Sig", "type": "normfactor"}
                        ],
                    }
                ],
            }
            for c in newspec['channels']
        ]
        newspec['channels'] = channels_with_signal

    return newspec
