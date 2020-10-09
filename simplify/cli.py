import click

import pyhf
pyhf.set_backend(pyhf.tensorlib, "minuit")

import json

from . import fitter
from . import yields
from . import model_utils
from . import simplified

@click.group()
@click.option('--debug/--no-debug', help="Enable debug mode",default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug

@cli.command()
@click.option('--input-file','-i', help="Input JSON likelihood file")
@click.option('--output-file','-o', help="Name of output JSON likelihood file")
@click.pass_context
def convert(ctx, input_file, output_file):

    click.echo("Loading input JSON")
    spec = json.load(open(input_file, "r"))

    click.echo("Getting model and data")
    model, data = model_utils.model_and_data(spec)

    click.echo("Bkg-only fit")
    fit_result = fitter.fit((model,data))

    # click.echo("Computing uncertainties")
    # stdevs = model_utils.calculate_stdev(model,fit_result.bestfit,fit_result.uncertainty,fit_result.corr_mat)

    click.echo("Getting psot-fit yields and uncertainties")
    ylds = yields.get_yields(spec, fit_result)

    click.echo("Building simplified likelihood")
    newspec = simplified.get_simplified_spec(spec, ylds, allowed_modifiers=["lumi"], prune_channels=[])

    with open(output_file, 'w') as ofile:
        json.dump(newspec, ofile, indent=4)
    click.echo('Debug is %s' % (ctx.obj['DEBUG'] and 'on' or 'off'))

@cli.command()
@click.pass_context
def validate(ctx):
    click.echo('Debug is %s' % (ctx.obj['DEBUG'] and 'on' or 'off'))

if __name__ == '__main__':
    cli(obj={})
