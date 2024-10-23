import logging
import sys


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO

    format_str = '%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Clear any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=date_format,
        stream=sys.stdout
    )
