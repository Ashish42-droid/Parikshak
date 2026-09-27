#!/usr/bin/env python3
"""CLI shim. The validator lives in parikshak/pdl/validator.py.

Kept at this path because it is the command in PLAN.md section 20 and in the
CI gate, and because a payload engineer authoring a procedure should not need
to know the package layout.

    python tools/validate_procedure.py procedures/my_experiment.yaml --strict
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parikshak.pdl.validator import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
