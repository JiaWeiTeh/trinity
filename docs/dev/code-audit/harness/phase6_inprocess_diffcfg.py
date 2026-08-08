#!/usr/bin/env python
"""In-process contamination across DIFFERENT configs — the case §7 left open.

`phase6_inprocess.py` runs the *same* config twice in one interpreter and passed
byte-identical. Its own scope note names the sharper hazard it does not cover:

    "The sharper documented hazard is a DIFFERENT config second, where a
     module-level cache keyed on the first config's data could serve stale
     values."

That is this probe. It compares config B against itself under two histories:

    solo : fresh interpreter, run B                      -> the uncontaminated truth
    pair : fresh interpreter, run A then B               -> B with A's leftovers

Both are launched as their own subprocess, so "solo" is genuinely uncontaminated
(CLAUDE.md: trinity leaks module-level global state in-process, which is why every
baseline in this audit is a separate process). Byte-identical B outputs => no
cross-config contamination on this path. Any divergence is the leak, and the first
differing key localises it.

    python docs/dev/code-audit/harness/phase6_inprocess_diffcfg.py

Writes: docs/dev/code-audit/data/phase6_diffcfg.md
"""

import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[4]
OUT = HERE.parents[1] / "data" / "phase6_diffcfg.md"
CFG_A = HERE.parent / "phase6_cfgA.param"
CFG_B = HERE.parent / "phase6_cfgB.param"

# Every active doc under docs/dev/ carries these four banners verbatim; the
# canonical wording lives in docs/dev/CLAUDE.md and test_docs_dev_conventions.py
# enforces their presence. This file is generated, so it has to emit them itself.
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

sys.path.insert(0, str(ROOT))

from phase6_inprocess import first_divergence, read_rows  # noqa: E402


def _run(param_path):
    """Run one config in THIS interpreter; return its dictionary.jsonl path."""
    from trinity._input.read_param import read_param
    from trinity.main import start_expansion

    params = read_param(str(param_path))
    start_expansion(params)
    return pathlib.Path(str(params["path2output"].value)) / "dictionary.jsonl"


def child(mode, dest):
    """Subprocess entry: produce config B's output under the given history."""
    if mode == "pair":
        _run(CFG_A)          # contaminant first, output discarded
    src = _run(CFG_B)
    shutil.copy(src, dest)
    return 0


def main():
    sys.path.insert(0, str(HERE.parent))
    results = {}
    for mode in ("solo", "pair"):
        dest = ROOT / "outputs" / f"phase6_diffcfg_{mode}.jsonl"
        print(f"== {mode}: launching subprocess ==")
        proc = subprocess.run(
            [sys.executable, str(HERE), "--child", mode, str(dest)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0 or not dest.exists():
            print(f"  FAILED (rc={proc.returncode})")
            print((proc.stderr or "")[-2000:])
            return 1
        rows = read_rows(dest)
        results[mode] = (dest, rows)
        print(f"  {mode}: {len(rows)} snapshots -> {dest.name}")

    (pa, a), (pb, b) = results["solo"], results["pair"]
    same_bytes = pa.read_bytes() == pb.read_bytes()
    div = first_divergence(a, b)

    lines = [
        "solo (fresh interpreter, B only) vs pair (A then B in one interpreter)",
        f"  solo : {len(a)} snapshots",
        f"  pair : {len(b)} snapshots",
        f"  byte-identical: {same_bytes}",
    ]
    if div is None:
        lines.append("  RESULT: identical - no cross-config contamination on this path")
        verdict = "PASS"
    else:
        i, k, va, vb = div
        lines.append(f"  RESULT: DIVERGE at snapshot {i}, key '{k}': solo={va!r} pair={vb!r}")
        verdict = "FAIL"
    print("\n" + "\n".join(lines))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "# Phase-6: in-process contamination across DIFFERENT configs\n\n"
        + BANNERS
        + f"\n**Status (generated by `{HERE.name}`):** \U0001f535 ACTIVE — closes the case\n"
        "`data/dynamic_verification.md` §7 left open: §7 ran the *same* config twice,\n"
        "this runs a **different** config first.\n\n"
        f"**Verdict: {verdict}**\n\n"
        "Config A (contaminant): `harness/phase6_cfgA.param` — mCloud 1e6, sfe 0.1, nCore 1e4\n"
        "Config B (subject): `harness/phase6_cfgB.param` — mCloud 1e5, sfe 0.3, schema-default nCore\n\n"
        "```\n" + "\n".join(lines) + "\n```\n\n"
        "Repro: `python docs/dev/code-audit/harness/phase6_inprocess_diffcfg.py`\n"
    )
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0 if div is None else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(child(sys.argv[2], pathlib.Path(sys.argv[3])))
    raise SystemExit(main())
