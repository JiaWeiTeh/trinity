#!/usr/bin/env python
"""Equivalence gate for the event-dispatch / isCollapse fix (NUM-02, S11-R-02).

CLAUDE.md rule 5: a change to an iterative path needs a FULL-RUN equivalence on
the stiff/edge regimes, in separate processes, at matched `t` -- a per-call check
is necessary but not sufficient.

The gate is deliberately NOT "bit-identical everywhere", because the fix is meant
to change one thing: `isCollapse` must stop being set on a shell that is
expanding at exit. So the bar is:

  G1a  every physics column is bit-identical to the pre-fix baseline
  G1b  `isCollapse` differs ONLY where the shell was expanding (v2 >= 0)

Anything else -- a moved trajectory, a changed row count, an isCollapse flip on a
contracting shell -- fails.

    python docs/dev/code-audit/harness/gate_event_dispatch_fix.py BASELINE_DIR

BASELINE_DIR holds the pre-fix dictionary.jsonl copies (maxr/momentum/diffcfg_solo).
Writes: docs/dev/code-audit/data/gate_event_dispatch_fix.md
"""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "gate_event_dispatch_fix.md"

# baseline file -> the post-fix run it must be compared against
CASES = [
    ("maxr.jsonl", "outputs/probe_iscollapse_maxr/dictionary.jsonl",
     "ends via large_radius while EXPANDING - the case the fix targets"),
    ("momentum.jsonl", "outputs/phase6_momentum/dictionary.jsonl",
     "ends via stop_t, reaches phase 2 - no terminal event fires"),
    ("diffcfg_solo.jsonl", "outputs/phase6_cfgB/dictionary.jsonl",
     "short control, ends via stop_t"),
]


def rows(path):
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def compare(base, new):
    """Return (physics_diffs, iscollapse_diffs) between two row lists."""
    physics, collapse = [], []
    for i, (a, b) in enumerate(zip(base, new)):
        for k in sorted(set(a) | set(b)):
            va, vb = a.get(k), b.get(k)
            if isinstance(va, float) and isinstance(vb, float):
                if va == vb or (math.isnan(va) and math.isnan(vb)):
                    continue
            elif va == vb:
                continue
            entry = (i, k, va, vb, b.get("v2"))
            (collapse if k == "isCollapse" else physics).append(entry)
    return physics, collapse


def main(baseline_dir):
    baseline_dir = pathlib.Path(baseline_dir)
    lines, verdicts = [], []

    for base_name, new_rel, why in CASES:
        bpath, npath = baseline_dir / base_name, ROOT / new_rel
        if not bpath.exists() or not npath.exists():
            lines.append(f"### {base_name} -- SKIPPED (missing file)")
            verdicts.append(False)
            continue

        base, new = rows(bpath), rows(npath)
        header = f"### `{base_name}` vs `{new_rel}`\n\n{why}\n"

        if len(base) != len(new):
            lines.append(header + f"\n**FAIL** - row count {len(base)} -> {len(new)}\n")
            verdicts.append(False)
            continue

        physics, collapse = compare(base, new)
        bad_collapse = [c for c in collapse if (c[4] is not None and c[4] < 0)]
        ok = not physics and not bad_collapse

        body = [
            header,
            f"- rows: {len(base)} (unchanged)",
            f"- physics columns differing: **{len(physics)}**"
            + ("" if not physics else f" -> {physics[:3]}"),
            f"- `isCollapse` rows differing: **{len(collapse)}**",
        ]
        for i, k, va, vb, v2 in collapse:
            body.append(
                f"    - row {i}: {va} -> {vb} at v2 = {v2:+.4f} pc/Myr"
                f" ({'EXPANDING - correct to clear' if v2 >= 0 else 'CONTRACTING - MUST NOT CHANGE'})"
            )
        body.append(f"\n**{'PASS' if ok else 'FAIL'}**\n")
        lines.append("\n".join(body))
        verdicts.append(ok)

    overall = "PASS" if all(verdicts) and verdicts else "FAIL"
    print("\n".join(lines))
    print(f"\nOVERALL: {overall}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "# Equivalence gate — event-dispatch / `isCollapse` fix\n\n"
        + BANNERS
        + f"\n**Status:** 🔵 ACTIVE — gate result for the `NUM-02` + `S11-R-02` fix.\n\n"
        "**Bar (set before editing, CLAUDE.md rule 5):** every physics column\n"
        "bit-identical to the pre-fix baseline, and `isCollapse` differing *only*\n"
        "where the shell was expanding (`v2 >= 0`) at exit.\n\n"
        f"**Result: {overall}**\n\n" + "\n".join(lines) + "\n\n"
        "Repro: re-run the three configs, then\n"
        "`python docs/dev/code-audit/harness/gate_event_dispatch_fix.py <baseline_dir>`\n"
    )
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0 if overall == "PASS" else 1


BANNERS = """\
> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
