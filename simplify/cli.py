import click

from .functions import workspaceFromJSON


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
    fullLH = workspaceFromJSON(input_file)

    click.echo('Debug is %s' % (ctx.obj['DEBUG'] and 'on' or 'off'))

@cli.command()
@click.pass_context
def validate(ctx):
    click.echo('Debug is %s' % (ctx.obj['DEBUG'] and 'on' or 'off'))

if __name__ == '__main__':
    cli(obj={})
