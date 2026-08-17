#!/usr/bin/env python3
"""Thin a TRINITY run directory down to every Nth snapshot.

Snapshots in ``dictionary.jsonl`` are independent JSON objects, one per line, so
keeping every Nth line leaves a file the reader loads unchanged — only the time
sampling is coarser. Profile arrays inside each kept snapshot are untouched, so
fidelity is exactly what the run produced.

The first and last snapshots are always kept, so the run still starts where it
started and ends where it ended.

    python thin_run.py SRC_RUN_DIR DEST_RUN_DIR --every 4
"""
import argparse
import json
import shutil
from pathlib import Path


def thin(src: Path, dest: Path, every: int) -> dict:
    lines = [ln for ln in (src / 'dictionary.jsonl').read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f'{src}/dictionary.jsonl is empty')

    keep = list(range(0, len(lines), every))
    if keep[-1] != len(lines) - 1:
        keep.append(len(lines) - 1)      # always keep the final state

    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'dictionary.jsonl').write_text('\n'.join(lines[i] for i in keep) + '\n')

    # metadata.json is what the reader rehydrates run constants from; the sidecar
    # .param and the human-readable summary travel with it for provenance.
    for name in ('metadata.json', 'metadata_humanreadable.txt'):
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    for extra in src.glob('*.param'):
        shutil.copy2(extra, dest / extra.name)

    first, last = json.loads(lines[0]), json.loads(lines[-1])
    return {
        'snapshots_in': len(lines),
        'snapshots_out': len(keep),
        'mb_in': (src / 'dictionary.jsonl').stat().st_size / 1e6,
        'mb_out': (dest / 'dictionary.jsonl').stat().st_size / 1e6,
        't_first': first['t_now'],
        't_last': last['t_now'],
        'phases': sorted({json.loads(lines[i])['current_phase'] for i in keep}),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src', type=Path)
    p.add_argument('dest', type=Path)
    p.add_argument('--every', type=int, default=4, help='keep every Nth snapshot (default 4)')
    a = p.parse_args()

    s = thin(a.src, a.dest, a.every)
    print(f"{s['snapshots_in']} -> {s['snapshots_out']} snapshots  "
          f"({s['mb_in']:.2f} -> {s['mb_out']:.2f} MB)")
    print(f"t = {s['t_first']:.4g} .. {s['t_last']:.4g} Myr   phases: {', '.join(s['phases'])}")
