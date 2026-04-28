#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract representative snapshots from a TRINITY simulation output.

Given a folder containing ``dictionary.jsonl``, this script writes 6 single-
snapshot ``.jsonl`` files into ``output/mockOutputs/<foldername>/``:

    1_begin.jsonl       first snapshot
    2_energy.jsonl      representative snapshot from the 'energy' phase
    3_implicit.jsonl    representative snapshot from the 'implicit' phase
    4_transition.jsonl  representative snapshot from the 'transition' phase
    5_momentum.jsonl    representative snapshot from the 'momentum' phase
    6_final.jsonl       last snapshot

For each phase, the second snapshot of that phase is chosen for stability.
If the second snapshot has already moved on to the next phase or marks
termination, the first snapshot of the phase is used instead. If a phase
never appears, that file is skipped with a warning.

Usage
-----
    python -m src._functions.extract_example_snapshots -F <folder>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src._output.trinity_reader import TrinityOutput


PHASES = [
    ('2_energy', 'energy'),
    ('3_implicit', 'implicit'),
    ('4_transition', 'transition'),
    ('5_momentum', 'momentum'),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dict_path(folder: Path) -> Path:
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    dict_path = folder / 'dictionary.jsonl'
    if not dict_path.is_file():
        raise FileNotFoundError(f"No dictionary.jsonl in: {folder}")
    return dict_path


def _is_terminated(snap_data: dict) -> bool:
    return bool(snap_data.get('EndSimulationDirectly')) or bool(snap_data.get('SimulationEndReason'))


def _pick_phase_index(output: TrinityOutput, phase: str) -> Optional[int]:
    phases = output.get('current_phase', as_array=False)
    i_begin = next((i for i, p in enumerate(phases) if p == phase), None)
    if i_begin is None:
        return None
    c = i_begin + 1
    if c >= len(output):
        return i_begin
    cand = output[c].data
    if cand.get('current_phase') != phase or _is_terminated(cand):
        return i_begin
    return c


def _write_snapshot(out_dir: Path, label: str, snap_data: dict) -> None:
    path = out_dir / f'{label}.jsonl'
    with open(path, 'w') as f:
        f.write(json.dumps(snap_data) + '\n')
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def extract(folder: Path) -> None:
    dict_path = _resolve_dict_path(folder)
    output = TrinityOutput.open(dict_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots in {dict_path}")

    out_dir = REPO_ROOT / 'output' / 'mockOutputs' / folder.resolve().name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source:  {dict_path}")
    print(f"Output:  {out_dir}")
    print(f"Snapshots: {len(output)}")

    _write_snapshot(out_dir, '1_begin', output[0].data)

    for label, phase in PHASES:
        idx = _pick_phase_index(output, phase)
        if idx is None:
            print(f"  skip  {label}: phase '{phase}' not found")
            continue
        _write_snapshot(out_dir, label, output[idx].data)

    _write_snapshot(out_dir, '6_final', output[len(output) - 1].data)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('-F', '--folder', required=True,
                        help="Folder containing dictionary.jsonl")
    args = parser.parse_args(argv)
    extract(Path(args.folder))
    return 0


if __name__ == '__main__':
    sys.exit(main())
