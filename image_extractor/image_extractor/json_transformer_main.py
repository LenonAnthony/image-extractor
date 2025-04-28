#!/usr/bin/env python
"""
Script to transform answer sheet JSON files from one format to another.
"""
import click
from pathlib import Path
from image_extractor.utils.json_transformer import transform_file, transform_directory


@click.group()
def cli():
    """Transform answer sheet JSON files to a different format."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--output', '-o', type=click.Path(), help='Output file path. If not provided, will add "_transformed" to the original filename.')
def transform_single(input_file, output):
    """Transform a single JSON file."""
    transform_file(input_file, output)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory. If not provided, will use the input directory.')
def transform_batch(input_dir, output_dir):
    """Transform all JSON files in a directory."""
    transform_directory(input_dir, output_dir)


if __name__ == '__main__':
    cli()
