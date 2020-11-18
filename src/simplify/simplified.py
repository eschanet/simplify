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
) -> pyhf.workspace:

    newspec = {
        'channels': [
            {
                'name': channel['name'],
                'samples': [
                    {
                        'name': 'Bkg',
                        # 'data': yields.yields[channel['name']],
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

    return newspec
