"""
trinity_to_cloudy.py — convert a TRINITY run into one or more CLOUDY input decks.

Canonical invocation::

    python -m src._output.cloudy.trinity_to_cloudy \\
        -F outputs/mockOutput/mockFullrun/ \\
        --age 0.1

Writes ``<run_dir>/cloudy/<model>_<index>_<phase>_t<age>myr.in`` plus a
sidecar ``.dlaw.txt`` with just the dlaw block, plus a copy of the bundled
``trinity_linelist.dat``. The ``<<<EDIT_ME>>>`` sentinel in the deck's
``table star`` line MUST be replaced by hand with the user's CLOUDY-compiled
SB99 atmosphere grid name before ``cloudy -r ...``.

Snapshot picker (mutually exclusive, exactly one required):
  --age MYR        cluster age (Myr since tSF) — picks closest snapshot
  --t-now MYR      raw simulation time (advanced)
  --index N        Nth snapshot, -1 = last
  --phase NAME [--pick first|last]
  --all            one deck per snapshot, plus manifest.json

See ``--help`` for full flag list.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src._output.cloudy.dlaw import DEFAULT_MIN_ROWS, DlawError
from src._output.cloudy.run_loader import RunBundle, RunLoadError, load_run
from src._output.cloudy.snapshot_to_deck import (
    DEFAULT_AGE_MAX_YR,
    DEFAULT_AGE_MIN_YR,
    SnapshotInvalid,
    snapshot_to_values,
)


TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "trinity_cloudy.in"
DEFAULT_LINELIST = TEMPLATE_DIR / "trinity_linelist.dat"
LINELIST_FILENAME = "trinity_linelist.dat"   # the name embedded into the deck

# A {{KEY}} placeholder. Word-boundary match means <<<EDIT_ME>>> is invisible
# to the renderer (passes through unchanged).
PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class UnsubstitutedPlaceholder(ValueError):
    """Raised when the rendered deck still contains {{KEY}} placeholders."""


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trinity_to_cloudy",
        description="Convert a TRINITY run directory into CLOUDY input decks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-F", "--folder", required=True, type=Path,
                        help="TRINITY run directory")

    pick = parser.add_mutually_exclusive_group()
    pick.add_argument("--age", type=float, metavar="MYR",
                      help="cluster age in Myr (canonical picker)")
    pick.add_argument("--t-now", type=float, metavar="MYR", dest="t_now",
                      help="raw simulation time (advanced)")
    pick.add_argument("--index", type=int, metavar="N",
                      help="Nth snapshot; -1 = last (advanced)")
    pick.add_argument("--phase", type=str, metavar="NAME",
                      help="phase name (advanced; pair with --pick)")
    pick.add_argument("--all", action="store_true",
                      help="one deck per snapshot; writes manifest.json")

    parser.add_argument("--pick", choices=("first", "last"), default="last",
                        help="for --phase: first or last snapshot in phase "
                             "(default: last)")

    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <RUN_DIR>/cloudy/)")
    parser.add_argument("--prefix", type=str, default=None,
                        help="filename prefix (default: auto-built from snapshot)")
    parser.add_argument("--template", type=Path, default=None,
                        help=f"template path (default: bundled {DEFAULT_TEMPLATE.name})")
    parser.add_argument("--linelist", type=Path, default=None,
                        help=f"line list path (default: bundled {DEFAULT_LINELIST.name})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print rendered deck to stdout; no writes")

    parser.add_argument("--abundances", type=str, default="HII region",
                        help='CLOUDY abundances directive value '
                             '(default "HII region")')
    parser.add_argument("--radius-out", type=float, default=None,
                        dest="radius_out_pc", metavar="PC",
                        help="extend dlaw past rShell into ambient ISM "
                             "(default: rShell)")
    parser.add_argument("--z-override", type=float, default=None,
                        dest="z_override", metavar="ZREL",
                        help="override summary.ZCloud")
    parser.add_argument("--age-min", type=float, default=DEFAULT_AGE_MIN_YR,
                        dest="age_min_yr", metavar="YR",
                        help=f"warn below this cluster age "
                             f"(default {DEFAULT_AGE_MIN_YR:.0e})")
    parser.add_argument("--age-max", type=float, default=DEFAULT_AGE_MAX_YR,
                        dest="age_max_yr", metavar="YR",
                        help=f"warn above this cluster age "
                             f"(default {DEFAULT_AGE_MAX_YR:.0e})")
    parser.add_argument("--hard-age-bounds", action="store_true",
                        help="promote the age-band warning to a hard error")
    parser.add_argument("--force", action="store_true",
                        help="proceed even if simulationEnd Status != SUCCESS")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS,
                        help=f"densify dlaw to >= N rows (default {DEFAULT_MIN_ROWS})")

    args = parser.parse_args(argv)

    # Exactly-one-picker enforcement (argparse mutex only enforces "at most one")
    n = sum([
        args.age is not None,
        args.t_now is not None,
        args.index is not None,
        args.phase is not None,
        args.all,
    ])
    if n == 0:
        parser.error(
            "exactly one snapshot picker is required: "
            "--age | --t-now | --index | --phase | --all"
        )

    if args.dry_run and args.all:
        parser.error("--all and --dry-run are mutually exclusive")

    return args


# --------------------------------------------------------------------------- #
# Snapshot picking
# --------------------------------------------------------------------------- #

@dataclass
class PickedSnapshot:
    index: int
    snap: Any   # TrinityOutput.Snapshot


def _pick_snapshots(bundle: RunBundle, args: argparse.Namespace) -> list[PickedSnapshot]:
    """Resolve the picker flags into a list of (index, snapshot) tuples."""
    if args.all:
        return [PickedSnapshot(index=s.index, snap=s) for s in bundle.output]

    if args.age is not None:
        target_t = args.age + float(bundle.metadata["tSF"])
        snap = bundle.output.get_at_time(target_t, mode="closest", quiet=True)
        return [PickedSnapshot(index=snap.index, snap=snap)]

    if args.t_now is not None:
        snap = bundle.output.get_at_time(args.t_now, mode="closest", quiet=True)
        return [PickedSnapshot(index=snap.index, snap=snap)]

    if args.index is not None:
        n = len(bundle.output)
        idx = args.index if args.index >= 0 else n + args.index
        if not (0 <= idx < n):
            raise SystemExit(
                f"index {args.index} out of range for {n} snapshots"
            )
        snap = bundle.output[idx]
        return [PickedSnapshot(index=snap.index, snap=snap)]

    if args.phase is not None:
        filtered = bundle.output.filter(phase=args.phase)
        if len(filtered) == 0:
            available = sorted({s.phase for s in bundle.output})
            raise SystemExit(
                f"no snapshots in phase {args.phase!r}; available: {available}"
            )
        which = -1 if args.pick == "last" else 0
        # filter() re-indexes from 0; map back to the original index by
        # round-tripping through get_at_time on the unfiltered output.
        target_t_now = filtered[which]["t_now"]
        snap = bundle.output.get_at_time(
            target_t_now, mode="closest", quiet=True,
        )
        return [PickedSnapshot(index=snap.index, snap=snap)]

    # _parse_args ensures we never reach here
    raise AssertionError("no picker selected (parser should have caught this)")


# --------------------------------------------------------------------------- #
# Status gate
# --------------------------------------------------------------------------- #

def _check_status(bundle: RunBundle, *, force: bool) -> None:
    status = bundle.end_state.get("status")
    if status == "SUCCESS":
        return
    if force:
        print(f"WARNING: run status is {status!r}; proceeding (--force)",
              file=sys.stderr)
        return
    raise SystemExit(
        f"run status is {status!r}; refusing to convert. Use --force to bypass."
    )


# --------------------------------------------------------------------------- #
# Prefix and output paths
# --------------------------------------------------------------------------- #

_UNSAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _build_prefix(
    args: argparse.Namespace,
    bundle: RunBundle,
    pick: PickedSnapshot,
    age_myr: float,
    phase: str,
) -> str:
    """
    Auto-build a filename-safe prefix:  <model>_<idx>_<phase>_t<age>myr
    Floats use "p" instead of "." so the prefix is shell- and CLOUDY-safe.
    """
    if args.prefix is not None:
        return _UNSAFE_PREFIX_RE.sub("_", args.prefix)
    age_part = f"t{age_myr:.4f}myr".replace(".", "p")
    raw = f"{bundle.model_name}_{pick.index}_{phase}_{age_part}"
    return _UNSAFE_PREFIX_RE.sub("_", raw)


def _resolve_out_dir(args: argparse.Namespace, bundle: RunBundle) -> Path:
    return args.out if args.out is not None else bundle.run_dir / "cloudy"


# --------------------------------------------------------------------------- #
# Template rendering
# --------------------------------------------------------------------------- #

def render_template(template_text: str, values: dict[str, Any]) -> str:
    """
    Substitute {{KEY}} placeholders. Raise UnsubstitutedPlaceholder on any
    {{KEY}} left after substitution. Sentinels not matching the {{KEY}}
    pattern (notably ``<<<EDIT_ME>>>``) pass through unchanged.
    """
    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        return match.group(0)
    out = PLACEHOLDER_RE.sub(sub, template_text)
    leftovers = sorted(set(PLACEHOLDER_RE.findall(out)))
    if leftovers:
        raise UnsubstitutedPlaceholder(
            f"unsubstituted placeholders in deck: {leftovers}"
        )
    return out


def _load_template(args: argparse.Namespace) -> str:
    p = args.template if args.template is not None else DEFAULT_TEMPLATE
    if not p.is_file():
        raise SystemExit(f"template not found: {p}")
    return p.read_text()


def _resolve_linelist(args: argparse.Namespace) -> Path:
    p = args.linelist if args.linelist is not None else DEFAULT_LINELIST
    if not p.is_file():
        raise SystemExit(f"linelist not found: {p}")
    return p


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        bundle = load_run(args.folder)
    except (RunLoadError, FileNotFoundError) as e:
        raise SystemExit(f"failed to load run: {e}")

    _check_status(bundle, force=args.force)

    picks = _pick_snapshots(bundle, args)
    out_dir = _resolve_out_dir(args, bundle)

    template_text = _load_template(args)
    linelist_path = _resolve_linelist(args)

    # Build deck(s)
    records: list[dict[str, Any]] = []
    for pick in picks:
        try:
            values = snapshot_to_values(
                pick.snap, bundle,
                z_override=args.z_override,
                radius_out_pc=args.radius_out_pc,
                age_min_yr=args.age_min_yr,
                age_max_yr=args.age_max_yr,
                hard_age_bounds=args.hard_age_bounds,
                min_rows=args.min_rows,
            )
            diag = values["_diagnostics"]
            prefix = _build_prefix(
                args, bundle, pick,
                age_myr=diag["age_myr"], phase=diag["phase"],
            )
            # Inject the CLI-derived substitution keys
            values["PREFIX"] = prefix
            values["ABUNDANCES"] = args.abundances
            values["LINELIST_FILENAME"] = LINELIST_FILENAME

            deck_text = render_template(template_text, values)

            if args.dry_run:
                # Single-snapshot dry run is enforced at parse time; just print.
                sys.stdout.write(deck_text)
                if not deck_text.endswith("\n"):
                    sys.stdout.write("\n")
                return 0

            _write_outputs(out_dir, prefix, deck_text, values["DLAW_BLOCK"])
            records.append({
                "index": pick.index,
                "t_now_myr": diag["t_now_myr"],
                "phase": diag["phase"],
                "age_myr": diag["age_myr"],
                "age_yr": diag["age_yr"],
                "n_dlaw_rows": diag["n_dlaw_rows"],
                "ambient_extended": diag["ambient_extended"],
                "z_used": diag["z_used"],
                "prefix": prefix,
                "deck": f"{prefix}.in",
                "dlaw_sidecar": f"{prefix}.dlaw.txt",
                "status": "ok",
                "reason": None,
            })
        except (SnapshotInvalid, DlawError, UnsubstitutedPlaceholder) as e:
            if not args.all:
                raise SystemExit(f"snapshot {pick.index}: {e}")
            records.append({
                "index": pick.index,
                "t_now_myr": pick.snap.get("t_now"),
                "phase": pick.snap.get("current_phase", "?"),
                "age_myr": None,
                "age_yr": None,
                "n_dlaw_rows": None,
                "ambient_extended": None,
                "z_used": None,
                "prefix": None,
                "deck": None,
                "dlaw_sidecar": None,
                "status": "skipped",
                "reason": str(e),
            })

    # Copy bundled linelist next to the decks (once per run)
    _copy_linelist(out_dir, linelist_path)

    if args.all:
        _write_manifest(out_dir, records)

    _print_summary(bundle, records, args, out_dir)
    return 0


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _write_outputs(out_dir: Path, prefix: str, deck_text: str,
                   dlaw_block: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    deck_path = out_dir / f"{prefix}.in"
    sidecar_path = out_dir / f"{prefix}.dlaw.txt"
    if not deck_text.endswith("\n"):
        deck_text += "\n"
    deck_path.write_text(deck_text)
    sidecar_path.write_text(dlaw_block + "\n")


def _copy_linelist(out_dir: Path, src: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / LINELIST_FILENAME
    # Only copy if missing or stale (avoids needless re-writes on --all)
    if not dst.is_file() or dst.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dst)


def _write_manifest(out_dir: Path, records: list[dict[str, Any]]) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


# --------------------------------------------------------------------------- #
# Closing summary (printed to stdout)
# --------------------------------------------------------------------------- #

_TODO = (
    "TODO:  edit <<<EDIT_ME>>> in the `table star` line — replace with the "
    "name of your CLOUDY-compiled SB99 atmosphere grid before "
    "`cloudy -r <prefix>`."
)


def _print_summary(
    bundle: RunBundle,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    if len(records) == 1:
        r = records[0]
        if r["status"] == "ok":
            print(
                f"Picked snapshot: index={r['index']}, phase={r['phase']}, "
                f"t_now={r['t_now_myr']:.4f} Myr"
            )
            print(f"                 cluster age = {r['age_yr']:.3e} yr")
            if args.age is not None:
                requested = args.age * 1.0e6
                delta = abs(r["age_yr"] - requested)
                print(
                    f"                 (requested {requested:.3e} yr, "
                    f"Δ={delta / 1e3:.2f} kyr)"
                )
            print(f"WROTE: {out_dir / r['deck']}")
            print(f"       {out_dir / r['dlaw_sidecar']}")
            print(f"       {out_dir / LINELIST_FILENAME}")
            print(_TODO)
        else:
            print(f"FAILED to convert snapshot {r['index']}: {r['reason']}")
    else:
        ok = sum(1 for r in records if r["status"] == "ok")
        skipped = len(records) - ok
        print(f"Converted {ok} snapshots ({skipped} skipped).")
        print(f"Output:   {out_dir}")
        print(f"Manifest: {out_dir / 'manifest.json'}")
        print(_TODO)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sys.exit(main())
