"""Enable `python -m elphgap` as an alias for the `elphgap` console script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
