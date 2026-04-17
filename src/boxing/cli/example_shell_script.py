import click

@click.command(context_settings={"show_default": True})
@click.argument("name", type=str, nargs=1)
@click.argument("input_paths", type=Path, nargs=-1)
@click.option("-p","--param", help="gimme parameter", default="bla")
@click.option("-vp","--verbose", help="be verbose", is_flag=True)
def df_concat(name, input_paths: list[Path], param="bla", verbose=False) -> None:
    print(name, input_paths, param, verbose)
