"""Rhizome CLI entry point."""

import click

from rhizome.cli.commands.ingest import ingest
from rhizome.cli.commands.traverse import traverse


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Rhizome: rhizomatic traversal of Wikipedia embeddings for writing aid."""
    pass


main.add_command(ingest)
main.add_command(traverse)


if __name__ == "__main__":
    main()
