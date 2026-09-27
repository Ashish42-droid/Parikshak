"""Console-script entry point: `parikshak-validate <file>... [--strict]`."""

import sys

from parikshak.pdl.validator import main

if __name__ == "__main__":
    sys.exit(main())
