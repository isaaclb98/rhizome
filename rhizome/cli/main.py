"""Rhizome CLI entry point."""

import os
from dotenv import load_dotenv

# Load .env file if it exists (API keys, Qdrant config, etc.)
load_dotenv()

import click

from rhizome.cli.commands.ingest import ingest
from rhizome.cli.commands.traverse import traverse
from rhizome.cli.commands.migrate import migrate_domain_field


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Rhizome: rhizomatic traversal of Wikipedia embeddings for writing aid."""
    pass


main.add_command(ingest)
main.add_command(traverse)
main.add_command(migrate_domain_field)


if __name__ == "__main__":
    main()
