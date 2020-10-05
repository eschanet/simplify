import pyhf
import numpy as np

from typing import Any, Dict, List, Tuple, Optional

import awkward1 as ak

from . import fitter
from . import validation

import logging
from . import logger
log = logging.getLogger(__name__)


def get_simplified_spec(
    spec: Dict[str, Any],
    yields: validation.Yields,
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
                        'data': np.zeros(yields.yields[channel]).size,
                        "modifiers": [
                            {
                                "data": None,
                                "name": "lumi",
                                "type": "lumi"
                            },
                            {
                                "data": None,
                                "name": "mu_Sig",
                                "type": "normfactor"
                            }
                        ],
                    },
                    {
                        'name': 'Signal',
                        'data': yields.yields[channel],
                        'modifiers': [
                            {
                                "data": {
                                    "hi_data": [
                                        yields.yields[channel] + yields.uncertainty[channel]
                                    ],
                                    "lo_data": [
                                        yields.yields[channel] - yields.uncertainty[channel]
                                    ]
                                },
                                "name": "totalError",
                                "type": "histosys"
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
                    'poi': 'mu_Sig'
                },
            }
            for measurement in self['measurements']
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

    return pyhf.workspace(newspec)
