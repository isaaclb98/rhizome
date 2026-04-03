"""Add the domain field to existing Qdrant chunks.

Run this once after upgrading to v0.4.0 if you have an existing
Qdrant collection created before the domain field was added.
"""

import click

from rhizome.config import get_config


@click.command()
@click.option(
    "--collection",
    type=str,
    default=None,
    help="Qdrant collection name (default: from QDRANT_COLLECTION env var)",
)
def migrate_domain_field(collection: str | None):
    """Migrate existing Qdrant chunks to add the domain field.

    This is required for collections created before v0.4.0.
    The migration queries PetScan to determine which Wikipedia domain
    each article belongs to, then patches the domain field in Qdrant.

    Example:
        rhizome migrate-domain-field --collection modernity-v1
    """
    from rhizome.migrations.add_domain_field import migrate_domain_field as _migrate

    collection = collection or get_config().qdrant_collection
    click.echo(f"Starting domain field migration for collection: {collection}")

    try:
        total, migrated = _migrate(collection)
        if migrated > 0:
            click.echo(f"Migration complete: {migrated}/{total} chunks updated")
        else:
            click.echo("No chunks needed migration")
    except Exception as e:
        click.echo(f"Migration failed: {e}", err=True)
        raise click.Abort()
