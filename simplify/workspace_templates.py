import pyhf
import numpy as np

from typing import Any, Dict, List, Tuple, Optional

import awkward1 as ak

from . import fitter

import logging
from . import logger
log = logging.getLogger(__name__)


class Simplified(dict):
    """Simplified likelihood object."""

    def __init__(self, regions, data, yields, uncertainties, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update(template)
