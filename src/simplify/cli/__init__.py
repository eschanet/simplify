import json
import logging
from typing import Any, List, Optional

import click
import pyhf

from .. import exceptions
from .. import fitter
from .. import model_tools
from .. import simplified
from .. import yields
from ..version import __version__

pyhf.set_backend(pyhf.tensorlib, "minuit")

log = logging.getLogger(__name__)


class OrderedGroup(click.Group):
    """A group that shows commands in the order they were added."""

    def list_commands(self, _: Any) -> List[str]:
        return list(self.commands.keys())


def _set_logging() -> None:
    """Sets log levels and format for CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
    )
    logging.getLogger("pyhf").setLevel(logging.WARNING)


@click.group(cls=OrderedGroup)
@click.version_option(version=__version__)
def simplify() -> None:
    """Top-level CLI entrypoint."""


@click.command()
@click.argument("workspace", default="-")
@click.option(
    '--output-file', '-o', default=None, help="Name of output JSON likelihood file"
)
@click.option(
    '--fixed-pars',
    '-f',
    default=[],
    multiple=True,
    help="Parameters to keep fixed at given value in fit (e.g. 'mu_SIG:1.0')",
)
@click.option(
    '--exclude-process',
    '-e',
    default=[],
    multiple=True,
    help="Process to be excluded in computation of fitted yields",
)
@click.option(
    '--dummy-signal/--no-dummy-signal',
    default=False,
    help="Output simplified likelihood with or without dummy signal",
)
@click.option(
    '--poi-name',
    default="lumi",
    help="Name of the POI. Defaults to lumi.",
)
def convert(
    workspace: str,
    dummy_signal: bool = False,
    poi_name: str = "lumi",
    output_file: Optional[str] = None,
    fixed_pars: Optional[List[str]] = None,
    exclude_process: Optional[List[str]] = None,
) -> None:

    fixed_pars = fixed_pars or []
    exclude_process = exclude_process or []

    # Read JSON spec
    with click.open_file(workspace, "r") as specstream:
        spec = json.load(specstream)

    if poi_name:
        try:
            spec['measurements'][0]["config"]["poi"] = poi_name
        except IndexError:
            raise exceptions.InvalidMeasurement(
                "The measurement index 0 is out of bounds."
            )

    # Get model and data
    model, data = model_tools.model_and_data(spec)

    # Set fixed params and run fit
    fixed_params = model.config.suggested_fixed()
    init_pars = model.config.suggested_init()
    if not fixed_pars:
        fixed_pars = []

    for (param, init) in [
        (param.split(':')[0], float(param.split(':')[1])) for param in fixed_pars
    ]:
        index = model_tools.get_parameter_names(model).index(param)
        fixed_params[index] = True
        init_pars[index] = float(init)

    fit_result = fitter.fit(model, data, init_pars=init_pars, fixed_pars=fixed_params)

    # Get yields
    ylds = yields.get_yields(spec, fit_result, exclude_process)

    # Hand yields to simplified LH builder and get simplified LH
    newspec = simplified.get_simplified_spec(
        spec, ylds, allowed_modifiers=[], prune_channels=[], include_signal=dummy_signal
    )

    if output_file is None:
        click.echo(json.dumps(newspec, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(newspec, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file}")


simplify.add_command(convert)
