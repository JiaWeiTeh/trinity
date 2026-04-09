#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Use ``run.py`` instead. Sweep mode is now auto-detected.

This script is kept for backward compatibility and delegates to run.py.
All sweep functionality (--workers, --dry-run, --yes, --verbose) is
available directly through run.py.

Example (new way):
    python run.py param/sweep.param --workers 4 --dry-run
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    print("NOTE: run_sweep.py is deprecated. Use run.py instead — "
          "sweep mode is auto-detected from parameter file content.\n")

    # Forward all arguments to run.py
    trinity_root = os.path.dirname(os.path.abspath(__file__))
    run_py = os.path.join(trinity_root, 'run.py')

    sys.exit(subprocess.call([sys.executable, run_py] + sys.argv[1:]))
