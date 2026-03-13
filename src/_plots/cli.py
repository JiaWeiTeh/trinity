# -*- coding: utf-8 -*-
"""
Shared CLI builder and dispatch logic for ``src._plots`` paper scripts.

Most paper_*.py scripts share the same argparse arguments and branching
logic (info / grid / single / help).  This module provides a builder
that creates the common parser and a dispatcher that runs the
appropriate mode.

Usage in a paper script::

    from src._plots.cli import build_parser, dispatch

    if __name__ == "__main__":
        dispatch(
            script_name="paper_betadelta.py",
            description="Plot TRINITY beta-delta",
            plot_from_path_fn=plot_from_path,
            plot_grid_fn=plot_grid,
        )

For scripts that need extra arguments::

    if __name__ == "__main__":
        parser = build_parser("paper_feedback.py", "Plot TRINITY feedback")
        parser.add_argument('--log-x', action='store_true',
                            help='Use log scale for x-axis')
        dispatch(
            parser=parser,
            plot_from_path_fn=plot_from_path,
            plot_grid_fn=plot_grid,
            pre_dispatch_fn=lambda args: globals().update(USE_LOG_X=True) if args.log_x else None,
        )
"""

import argparse
from typing import Callable, Optional

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._output.trinity_reader import info_simulations


def build_parser(
    script_name: str,
    description: str,
) -> argparse.ArgumentParser:
    """Create the standard paper-script argument parser.

    Returns the parser so callers can add extra arguments before
    calling ``dispatch(parser=...)``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
  # Single simulation
  python {script_name} 1e7_sfe020_n1e4
  python {script_name} /path/to/outputs/1e7_sfe020_n1e4

  # Grid plot from folder
  python {script_name} --folder /path/to/my_experiment/
  python {script_name} -F /path/to/simulations/ -n 1e4

  # Filter by cloud mass and/or SFE
  python {script_name} -F /path/to/simulations/ --mCloud 1e6 1e7
  python {script_name} -F /path/to/simulations/ --sfe 001 010 020

  # Scan folder for available parameters
  python {script_name} -F /path/to/simulations/ --info
        """,
    )
    parser.add_argument(
        "data",
        nargs="?",
        default=None,
        help="Data input: folder name, folder path, or file path (for single simulation)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save output figures (default: fig/)",
    )
    parser.add_argument(
        "--folder",
        "-F",
        default=None,
        help=(
            "Create grid plot from all simulations in folder. "
            "Auto-organizes by mCloud (rows) and SFE (columns)."
        ),
    )
    parser.add_argument(
        "--nCore",
        "-n",
        default=None,
        help=(
            'Filter simulations by cloud density (e.g., "1e4", "1e3"). '
            "If not specified, generates one PDF per density found."
        ),
    )
    parser.add_argument(
        "--mCloud",
        nargs="+",
        default=None,
        help="Filter simulations by cloud mass (e.g., --mCloud 1e6 1e7).",
    )
    parser.add_argument(
        "--sfe",
        nargs="+",
        default=None,
        help="Filter simulations by SFE (e.g., --sfe 001 010).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Scan folder and print available mCloud, SFE, and nCore values.",
    )
    return parser


def _print_info(folder: str) -> None:
    """Print available simulation parameters in *folder*."""
    info = info_simulations(folder)
    print("=" * 50)
    print(f"Simulation parameters in: {folder}")
    print("=" * 50)
    print(f"  Total simulations: {info['count']}")
    print(f"  mCloud values: {info['mCloud']}")
    print(f"  SFE values: {info['sfe']}")
    print(f"  nCore values: {info['ndens']}")


def dispatch(
    *,
    plot_from_path_fn: Callable,
    plot_grid_fn: Callable,
    script_name: str = "",
    description: str = "",
    parser: Optional[argparse.ArgumentParser] = None,
    pre_dispatch_fn: Optional[Callable] = None,
) -> None:
    """Parse CLI arguments and run the appropriate mode.

    Parameters
    ----------
    plot_from_path_fn : callable(data_input, output_dir)
        Function for single-simulation mode.
    plot_grid_fn : callable(folder, output_dir, ndens_filter, mCloud_filter, sfe_filter)
        Function for grid mode.
    script_name, description :
        Passed to ``build_parser`` when *parser* is ``None``.
    parser :
        Pre-built parser (use when extra arguments are needed).
    pre_dispatch_fn :
        Optional callback ``fn(args)`` invoked after parsing but
        before dispatching.  Useful for setting globals from extra
        flags (e.g. ``USE_LOG_X``).
    """
    if parser is None:
        parser = build_parser(script_name, description)

    args = parser.parse_args()

    if pre_dispatch_fn is not None:
        pre_dispatch_fn(args)

    if args.info:
        if not args.folder:
            parser.print_help()
            print("\nError: --info requires --folder to be specified.")
        else:
            _print_info(args.folder)
    elif args.folder:
        plot_grid_fn(
            args.folder,
            args.output_dir,
            ndens_filter=args.nCore,
            mCloud_filter=args.mCloud,
            sfe_filter=args.sfe,
        )
    elif args.data:
        plot_from_path_fn(args.data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either --folder or a data path.")
