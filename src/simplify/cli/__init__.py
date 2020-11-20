import json
import logging
from typing import Any, KeysView

import click
import pyhf

from .. import fitter
from .. import model_tools
from .. import simplified
from .. import yields
from ..version import __version__

pyhf.set_backend(pyhf.tensorlib, "minuit")


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
@click.option('--input-file', '-i', help="Input JSON likelihood file")
@click.option('--output-file', '-o', help="Name of output JSON likelihood file")
def convert(input_file: str, output_file: str) -> None:

    click.echo("Loading input JSON")
    spec = json.load(open(input_file, "r"))

    click.echo("Getting model and data")
    model, data = model_tools.model_and_data(spec)

    click.echo("Bkg-only fit")
    fit_result = fitter.fit(spec)

    click.echo("Getting post-fit yields and uncertainties")
    ylds = yields.get_yields(spec, fit_result)

    click.echo("Building simplified likelihood")
    newspec = simplified.get_simplified_spec(
        spec, ylds, allowed_modifiers=["lumi"], prune_channels=[]
    )

    with open(output_file, 'w') as ofile:
        json.dump(newspec, ofile, indent=4)


simplify.add_command(convert)
