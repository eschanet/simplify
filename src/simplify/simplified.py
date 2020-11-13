import pyhf
import numpy as np
import copy

from typing import Any, Dict, List, Tuple, Optional

import awkward1 as ak

from . import fitter
from . import yields

import logging
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
                        'data': ylds.yields[channel['name']].sum(axis=0).flatten().tolist(),
                        "modifiers": [
                            {
                                "data": {
                                    "hi_data": (ylds.yields[channel['name']].sum(axis=0) + ak.to_numpy(ylds.uncertainties[channel['name']])).flatten().tolist(), # flatten to make flat list. Array is 1D anyway already
                                    "lo_data": (ylds.yields[channel['name']].sum(axis=0) - ak.to_numpy(ylds.uncertainties[channel['name']])).flatten().tolist(), # flatten to make flat list. Array is 1D anyway already
                                },
                                "name": "totalError",
                                "type": "histosys"
                            }
                        ],
                    }#,
                    # {
                    #     'name': 'Signal',
                    #     'data': np.zeros(ylds.yields[channel['name']].sum(axis=0).size).tolist(),
                    #     'modifiers': [
                    #         {
                    #             "data": None,
                    #             "name": "lumi",
                    #             "type": "lumi"
                    #         },
                    #         {
                    #             "data": None,
                    #             "name": "mu_Sig",
                    #             "type": "normfactor"
                    #         }
                    #     ],
                    # }
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
