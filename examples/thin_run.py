#!/usr/bin/env python3
"""Prepare a TRINITY run directory for shipping: de-duplicate, sort, thin.

Three lossless steps, in order:

1. **De-duplicate.** A run can contain the same snapshot more than once — the
   snapshot buffer writes overlapping chunks, and a long run may repeat close to
   half its lines. Duplicates are dropped on ``t_now``.
2. **Sort by time.** Snapshots are written in buffer-flush order, which is *not*
   time order. Sorting makes the file chronological, so a consumer can plot it
   directly instead of getting a zig-zag. It also restores the phase sequence to
   the forward-only order the solver actually follows.
3. **Thin.** Keep every Nth remaining snapshot, so the run is small enough to
   commit. Profile arrays inside a kept snapshot are untouched. The first and
   last snapshots are always kept, so the run still starts and ends where it did.

Steps 1 and 2 discard nothing and reorder nothing that carries meaning; only
step 3 drops data.

    python thin_run.py SRC_RUN_DIR DEST_RUN_DIR --every 4
"""
import argparse
import json
import shutil
from pathlib import Path


def prepare(src: Path, dest: Path, every: int) -> dict:
    raw = [ln for ln in (src / 'dictionary.jsonl').read_text().splitlines() if ln.strip()]
    if not raw:
        raise SystemExit(f'{src}/dictionary.jsonl is empty')

    # 1. de-duplicate on t_now, keeping the first occurrence
    seen, unique = set(), []
    for ln in raw:
        t = json.loads(ln)['t_now']
        if t not in seen:
            seen.add(t)
            unique.append((t, ln))

    # 2. sort chronologically
    unique.sort(key=lambda pair: pair[0])

    # 3. thin, always keeping the first and last
    keep = list(range(0, len(unique), every))
    if keep[-1] != len(unique) - 1:
        keep.append(len(unique) - 1)

    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'dictionary.jsonl').write_text(
        '\n'.join(unique[i][1] for i in keep) + '\n')

    # metadata.json is what the reader rehydrates run constants from; the sidecar
    # .param and the human-readable summary travel with it for provenance.
    for name in ('metadata.json', 'metadata_humanreadable.txt'):
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    for extra in src.glob('*.param'):
        shutil.copy2(extra, dest / extra.name)

    phases = [json.loads(unique[i][1])['current_phase'] for i in keep]
    order = {'energy': 0, 'implicit': 1, 'transition': 2, 'momentum': 3}
    ranks = [order[p] for p in phases]
    return {
        'raw': len(raw),
        'deduped': len(unique),
        'kept': len(keep),
        'mb_in': (src / 'dictionary.jsonl').stat().st_size / 1e6,
        'mb_out': (dest / 'dictionary.jsonl').stat().st_size / 1e6,
        't_first': unique[keep[0]][0],
        't_last': unique[keep[-1]][0],
        'phases': list(dict.fromkeys(phases)),
        'forward_only': all(b >= a for a, b in zip(ranks, ranks[1:])),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src', type=Path)
    p.add_argument('dest', type=Path)
    p.add_argument('--every', type=int, default=4, help='keep every Nth snapshot (default 4)')
    a = p.parse_args()

    s = prepare(a.src, a.dest, a.every)
    dropped = s['raw'] - s['deduped']
    print(f"{s['raw']} snapshots read")
    if dropped:
        print(f"  -{dropped} duplicates removed ({100 * dropped / s['raw']:.0f}% of the file)")
    print(f"  sorted by t_now, then kept every {a.every}th")
    print(f"  -> {s['kept']} snapshots  ({s['mb_in']:.2f} -> {s['mb_out']:.2f} MB)")
    print(f"t = {s['t_first']:.4g} .. {s['t_last']:.4g} Myr")
    print(f"phases: {' -> '.join(s['phases'])}"
          f"   {'(forward-only)' if s['forward_only'] else '*** NOT forward-only ***'}")
