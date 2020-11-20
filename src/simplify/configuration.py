import logging
import pathlib
from typing import Any, Dict, Union

import yaml


log = logging.getLogger(__name__)


def load(file_path_string: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads, validates and returns a cfg file from path."""

    file_path = pathlib.Path(file_path_string)
    log.info(f"opening config file {file_path}")
    config = yaml.safe_load(file_path.read_text())
    # validate(config)
    return config


def print_overview(config: Dict[str, Any]) -> None:
    """Prints a compact summary of a config file."""

    log.info("the config contains:")
    if "Samples" in config.keys():
        log.info(f"  {len(config['Samples'])} Sample(s)")
    log.info(f"  {len(config['Regions'])} Regions(s)")
    if "NormFactors" in config.keys():
        log.info(f"  {len(config['NormFactors'])} NormFactor(s)")
    if "Systematics" in config.keys():
        log.info(f"  {len(config['Systematics'])} Systematic(s)")


def get_region_dict(config: Dict[str, Any], region_name: str) -> Dict[str, Any]:
    """Returns the dictionary for a region with the given name."""
    regions = [reg for reg in config["Regions"] if reg["Name"] == region_name]
    if len(regions) == 0:
        raise ValueError(f"region {region_name} not found in config")
    if len(regions) > 1:
        log.error(f"found more than one region with name {region_name}")
    return regions[0]
