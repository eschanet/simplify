import json
import logging
from typing import Any, KeysView, Optional

import click
import pyhf

from .. import fitter
from .. import model_tools
from .. import simplified
from .. import yields
from ..version import __version__

pyhf.set_backend(pyhf.tensorlib, "minuit")

log = logging.getLogger(__name__)


class OrderedGroup(click.Group):
    """A group that shows commands in the order they were added."""

    def list_commands(self, _: Any) -> KeysView[str]:
        return self.commands.keys()


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
def convert(workspace: str, output_file: Optional[str] = None) -> None:

    log.debug("Loading input")
    with click.open_file(workspace, "r") as specstream:
        spec = json.load(specstream)

    log.debug("Getting model and data")
    model, data = model_tools.model_and_data(spec)

    log.debug("Bkg-only fit")
    fit_result = fitter.fit(spec)

    log.debug("Getting post-fit yields and uncertainties")
    ylds = yields.get_yields(spec, fit_result)

    log.debug("Building simplified likelihood")
    newspec = simplified.get_simplified_spec(
        spec, ylds, allowed_modifiers=["lumi"], prune_channels=[]
    )

    if output_file is None:
        click.echo(json.dumps(newspec, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(newspec, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file}")


simplify.add_command(convert)
