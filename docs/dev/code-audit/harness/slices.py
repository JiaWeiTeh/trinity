"""The audit's disjoint slice partition, and the per-slice Lens A/B inputs.

Every .py under trinity/ belongs to exactly one slice (asserted below), so no
file is reviewed twice and none is missed. Slices follow physics coherence, not
file size.

    python docs/dev/code-audit/harness/slices.py <outdir>

writes, per slice: <outdir>/<slice>/code/...      comment-stripped source (Lens A)
                   <outdir>/<slice>/prose.md      comments + docstrings only (Lens B)
                   <outdir>/<slice>/signatures.md def/class signatures only (Lens C)
"""

import ast
import pathlib
import sys

# slice id -> (title, tier, [files])
# tier "physics" gets all three lenses; "infra" gets A + B only.
SLICES = {
    "S1_units_helpers": ("Units & shared helpers", "physics", [
        "_functions/unit_conversions.py", "_functions/operations.py",
        "_functions/cluster.py", "_functions/simplify.py",
        "_functions/logging_setup.py", "_functions/extract_example_snapshots.py",
        "_functions/__init__.py",
    ]),
    "S2_cloud": ("Cloud properties (PL / Bonnor-Ebert profiles)", "physics", [
        "cloud_properties/density_profile.py", "cloud_properties/mass_profile.py",
        "cloud_properties/powerLawSphere.py", "cloud_properties/bonnorEbertSphere.py",
        "cloud_properties/initial_profile.py", "cloud_properties/validate_gmc.py",
        "cloud_properties/__init__.py",
    ]),
    "S3_phase0": ("Phase 0 — cloud + free-streaming initialisation", "physics", [
        "phase0_init/get_InitCloudProp.py", "phase0_init/get_InitPhaseParam.py",
        "phase0_init/__init__.py",
    ]),
    "S4_phase1_energy": ("Phase 1 — energy-driven (Weaver) phase", "physics", [
        "phase1_energy/energy_phase_ODEs.py", "phase1_energy/run_energy_phase.py",
        "phase1_energy/__init__.py",
    ]),
    "S5a_betadelta": ("Phase 1b — the beta/delta implicit solve", "physics", [
        "phase1b_energy_implicit/get_betadelta.py",
    ]),
    "S5b_implicit_runner": ("Phase 1b — implicit-phase runner", "physics", [
        "phase1b_energy_implicit/run_energy_implicit_phase.py",
        "phase1b_energy_implicit/__init__.py",
    ]),
    "S6_transition_momentum": ("Phase 1c transition + phase 2 momentum", "physics", [
        "phase1c_transition/run_transition_phase.py", "phase1c_transition/__init__.py",
        "phase2_momentum/run_momentum_phase.py", "phase2_momentum/__init__.py",
    ]),
    "S7_bubble": ("Bubble structure & luminosity", "physics", [
        "bubble_structure/bubble_luminosity.py", "bubble_structure/get_bubbleParams.py",
        "bubble_structure/__init__.py",
    ]),
    "S8_shell": ("Shell structure & shell ODE", "physics", [
        "shell_structure/shell_structure.py", "shell_structure/get_shellODE.py",
        "shell_structure/__init__.py",
    ]),
    "S9_cooling": ("Cooling — CIE, non-CIE, net curve", "physics", [
        "cooling/net_coolingcurve.py", "cooling/CIE/read_coolingcurve.py",
        "cooling/non_CIE/read_cloudy.py", "cooling/__init__.py",
        "cooling/CIE/__init__.py", "cooling/non_CIE/__init__.py",
    ]),
    "S10_sps": ("SPS tables & feedback update", "physics", [
        "sps/read_sps.py", "sps/sps_columns.py", "sps/update_feedback.py",
        "sps/__init__.py",
    ]),
    "S11_orchestration": ("Orchestration & phase events", "physics", [
        "main.py", "phase_general/phase_events.py", "phase_general/__init__.py",
        "__init__.py",
    ]),
    "S12a_input_config": ("Input — config, schema, registry", "infra", [
        "_input/dictionary.py", "_input/registry.py", "_input/read_param.py",
        "_input/param_spec.py", "_input/errors.py", "_input/fkappa_auto.py",
        "_input/__init__.py",
    ]),
    "S12b_input_sweep": ("Input — sweep expansion & job emission", "infra", [
        "_input/sweep_parser.py", "_input/sweep_jobs.py", "_input/sweep_runner.py",
    ]),
    "S13a_output_core": ("Output — reader, snapshots, terminal, metadata", "infra", [
        "_output/trinity_reader.py", "_output/show_run.py", "_output/simulation_end.py",
        "_output/terminal_prints.py", "_output/header.py", "_output/run_constants.py",
        "_output/_metadata_io.py", "_output/__init__.py",
    ]),
    "S13b_output_cloudy": ("Output — CLOUDY export", "infra", [
        "_output/cloudy/trinity_to_cloudy.py", "_output/cloudy/snapshot_to_deck.py",
        "_output/cloudy/run_loader.py", "_output/cloudy/dlaw.py",
        "_output/cloudy/__init__.py",
    ]),
    "S14_analysis": ("In-package analysis helpers", "infra", [
        "_analysis/check_yesno.py", "_analysis/__init__.py",
    ]),
}

ROOT = pathlib.Path(__file__).resolve().parents[4]
PKG = ROOT / "trinity"


def check_partition():
    """Every .py under trinity/ in exactly one slice — no gaps, no overlap."""
    on_disk = {p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")}
    assigned = [f for _, _, files in SLICES.values() for f in files]
    dupes = {f for f in assigned if assigned.count(f) > 1}
    assigned = set(assigned)
    assert not dupes, f"assigned to >1 slice: {sorted(dupes)}"
    assert not on_disk - assigned, f"unassigned files: {sorted(on_disk - assigned)}"
    assert not assigned - on_disk, f"assigned but missing: {sorted(assigned - on_disk)}"
    return len(on_disk)


def rows_sig(src):
    """Signature lines only: `def`/`class` headers and module-level constant names.

    Lens C must derive what a function *should* compute from its interface and the
    literature, so it gets names, parameters and annotations — never a body, never a
    numeric literal (a stale constant would anchor the derivation it exists to check).
    """
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            out.append((node.lineno, f"def {node.name}({ast.unparse(node.args)}){ret}"))
        elif isinstance(node, ast.ClassDef):
            out.append((node.lineno, f"class {node.name}"))
    for node in ast.parse(src).body:
        for tgt in getattr(node, "targets", []):
            if isinstance(tgt, ast.Name):
                out.append((node.lineno, f"{tgt.id} = ..."))
    return sorted(out)


def main(out_dir):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from extract_claims import rows_prose
    from strip_comments import strip

    n = check_partition()
    out_dir = pathlib.Path(out_dir)
    for sid, (title, tier, files) in SLICES.items():
        base = out_dir / sid
        prose = [f"# {title} — prose only ({tier} tier)\n"]
        sigs = [f"# {title} — signatures only ({tier} tier)\n"]
        for rel in files:
            src = (PKG / rel).read_text()
            dst = base / "code" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(strip(src))
            if lines := rows_sig(src):
                sigs.append(f"\n## trinity/{rel}\n")
                sigs += [f"- `L{ln}` `{text}`" for ln, text in lines]
            rows = rows_prose(f"trinity/{rel}", src)
            if not rows:
                continue
            prose.append(f"\n## trinity/{rel}\n")
            prose += [
                f"- `L{r['line']}-{r['end_line']}` **{r['kind']}"
                f"{' ' + r['owner'] if r['owner'] else ''}** — {r['text']}"
                for r in rows
            ]
        base.mkdir(parents=True, exist_ok=True)
        (base / "prose.md").write_text("\n".join(prose) + "\n")
        (base / "signatures.md").write_text("\n".join(sigs) + "\n")
    print(f"{n} files -> {len(SLICES)} slices in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
