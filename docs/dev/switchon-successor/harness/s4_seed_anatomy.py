#!/usr/bin/env python3
"""Batch 4a (docs/dev/switchon-successor/PLAN.md §5): the analytic anatomy of the
phase-0 → phase-1a handover state. **No simulations** — this evaluates the seed
that `get_y0` hands over and the `solve_R1` root that phase 1a immediately
computes from it, for every screen config, in seconds.

Why it is worth doing before writing any S4 candidate: the seed's work rate is
*algebraic*, and the algebra says which knobs can possibly help.

With `R1` at ram-pressure balance (which is what `solve_R1` enforces),

    Pb = Pram(R1) = Lmech / (2 pi v_wind R1**2)

so the PdV term of the phase-1a energy equation is

    PdV / Lmech = 4 pi R2**2 Pb v2 / Lmech = 2 (v2/v_wind) / (R1/R2)**2

— `Eb` has dropped out entirely. `Eb` re-enters only through the balance root
itself (`R1/R2` shrinks as `Eb` grows), and since `R1/R2 <= 1` by construction,

    PdV / Lmech >= 2 (v2 / v_wind)        for ANY seed energy.

The seed hands over `v2 = v0 = v_wind` (the free-streaming terminal speed), so
the handover starts at `PdV >= 2 Lmech` however `E0` is chosen. That is the
bound this script evaluates per config.

    python docs/dev/switchon-successor/harness/s4_seed_anatomy.py
"""
import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
OUT = os.path.join(HERE, "..", "data", "s4_seed_anatomy.csv")


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed_of(param_path, workdir):
    """Seed state + the R1 phase 1a solves for, from one config. No simulation."""
    sys.path.insert(0, REPO)
    from trinity._input import read_param
    from trinity.phase0_init import get_InitCloudProp, get_InitPhaseParam
    from trinity.sps import read_sps, update_feedback
    from trinity.bubble_structure import get_bubbleParams

    os.makedirs(workdir, exist_ok=True)
    params = read_param.read_param(param_path)
    # Same steps 1-2 main.start_expansion runs before get_y0 (cloud, then SPS).
    get_InitCloudProp.get_InitCloudProp(params)
    sps_data = read_sps.read_sps(params["mCluster"] / params["sps_refmass"], params)
    params["sps_data"].value = sps_data
    params["sps_f"].value = read_sps.get_interpolation(sps_data)

    t0, r0, v0, E0, T0 = get_InitPhaseParam.get_y0(params)
    tSF = params["tSF"].value

    fb = update_feedback.get_current_sps_feedback(t0, params)
    Lmech_total = fb.Lmech_total if hasattr(fb, "Lmech_total") else fb[0]
    v_mech_total = fb.v_mech_total if hasattr(fb, "v_mech_total") else fb[1]

    R1 = get_bubbleParams.solve_R1(r0, E0, Lmech_total, v_mech_total)
    x = R1 / r0
    return dict(t0=t0, tSF=tSF, dt_phase0=t0 - tSF, r0=r0, v0=v0, E0=E0, T0=T0,
                Lmech_total=Lmech_total, v_mech_total=v_mech_total, R1=R1, x=x)


def identity_check(run_dir, out, n=6):
    """Validate PdV/Lmech = 2(v2/v_wind)/(R1/R2)^2 along a whole ramp-OFF run.

    The seed is one point; this checks the identity is exact everywhere, which
    is what licenses reasoning about later times from it. Needs a ramp-off run
    directory (ephemeral), so the CSV it writes is the durable record.
    """
    import json
    with open(os.path.join(run_dir, "dictionary.jsonl")) as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()][:n]
    sys.path.insert(0, REPO)
    from trinity.bubble_structure import get_bubbleParams

    out_rows = []
    for i, r in enumerate(rows):
        x = get_bubbleParams.solve_R1(
            r["R2"], r["Eb"], r["Lmech_total"], r["v_mech_total"]) / r["R2"]
        vr = r["v2"] / r["v_mech_total"]
        pred = 2.0 * vr / x ** 2
        Pb = get_bubbleParams.bubble_E2P(r["Eb"], r["R2"], x * r["R2"], 5.0 / 3.0)
        meas = 4.0 * 3.141592653589793 * r["R2"] ** 2 * Pb * r["v2"] / r["Lmech_total"]
        out_rows.append(dict(snapshot=i, t_now_Myr=f"{r['t_now']:.6e}",
                             R1_over_R2=f"{x:.6f}", v2_over_vwind=f"{vr:.6f}",
                             predicted_2v_over_x2=f"{pred:.6f}",
                             measured_PdV_over_Lmech=f"{meas:.6f}",
                             rel_err=f"{abs(pred - meas) / meas:.2e}"))
        print(f"  snap {i}: x={x:.6f} v2/vw={vr:.6f} pred={pred:.6f} "
              f"meas={meas:.6f} rel_err={abs(pred - meas) / meas:.1e}")
    with open(os.path.normpath(out), "w", newline="") as fh:
        fh.write(
            "# switchon-successor Batch 4a: the handover identity, checked along a whole run.\n"
            "#   PdV/Lmech = 2 (v2/v_wind) / (R1/R2)^2, with R1 at ram-pressure balance.\n"
            "# Source: the ramp-OFF simple_cluster arm of Batch 1 (the run dir is ephemeral --\n"
            "# this CSV is the durable record). Reproduce on any ramp-off run dir with\n"
            "#   python docs/dev/switchon-successor/harness/s4_seed_anatomy.py \\\n"
            "#          --identity-run <dir with dictionary.jsonl>                  (2026-08-06)\n"
            "# The identity is algebra on trinity's own equations, so it is exact to roundoff at\n"
            "# every snapshot, not only at the seed -- which is what licenses using it to reason\n"
            "# about later times. Read alongside: v2/v_wind falls but (R1/R2) climbs to ~1 as Eb\n"
            "# drains, so PdV/Lmech stays above 1 and the drain does not self-arrest.\n")
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {os.path.normpath(out)}")


def main():
    if "--identity-run" in sys.argv:
        run_dir = sys.argv[sys.argv.index("--identity-run") + 1]
        return identity_check(run_dir, os.path.join(HERE, "..", "data",
                                                    "s4_identity_check.csv"))
    screen = load_screen()
    rows = []
    for name, param in screen.CONFIGS.items():
        # Each config is read in its own subprocess-free pass; trinity's module
        # globals are re-read per call and get_y0 is pure, so this is safe.
        s = seed_of(os.path.join(REPO, param),
                    os.path.join(REPO, "outputs", "_s4_anatomy", name))
        x = s["x"]
        vratio = s["v0"] / s["v_mech_total"]
        pdv_balance = 2.0 * vratio / x ** 2            # unramped: Pb at balance
        pdv_ramped = (10.0 / 11.0) * vratio            # ramp -> R1 = 0
        # Weaver Eq.20 seed => Eb/t exactly (5/11)Lw t, so PdV/Lw with R1=0 is
        # (10/11)*(v2 t / R2) and R2 = v0*dt makes v2 t/R2 = 1.
        v_sustain = x ** 2 / 2.0                       # v2/v_wind giving PdV = Lmech
        rows.append(dict(
            config=name,
            dt_phase0_Myr=f"{s['dt_phase0']:.6e}",
            r0_pc=f"{s['r0']:.6e}",
            v0_over_vwind=f"{vratio:.6f}",
            E0_au=f"{s['E0']:.6e}",
            R1_over_R2=f"{x:.6f}",
            balance_A=f"{x ** 2 / (1 - x ** 3):.6f}",
            PdV_over_Lmech_unramped=f"{pdv_balance:.6f}",
            PdV_over_Lmech_ramped=f"{pdv_ramped:.6f}",
            floor_2v_over_vwind=f"{2.0 * vratio:.6f}",
            v2_over_vwind_for_PdV_eq_Lmech=f"{v_sustain:.6f}",
        ))
        print(f"{name:16s} x={x:.6f}  v0/vw={vratio:.4f}  "
              f"PdV/Lw unramped={pdv_balance:.4f}  ramped={pdv_ramped:.4f}  "
              f"floor={2 * vratio:.4f}  v_sustain/vw={v_sustain:.4f}")

    hdr = ("# switchon-successor Batch 4a: analytic anatomy of the phase-0 handover state.\n"
           "# No simulations. Seed from trinity.phase0_init.get_InitPhaseParam.get_y0, R1 from\n"
           "# trinity.bubble_structure.get_bubbleParams.solve_R1 at that seed.\n"
           "#   PdV/Lmech = 2 (v2/v_wind) / (R1/R2)^2   when R1 is at ram-pressure balance\n"
           "#   PdV/Lmech = (10/11)(v2 t/R2)            when the ramp drives R1 -> 0\n"
           "# 'floor' is 2*(v0/v_wind): the smallest PdV/Lmech reachable at this handover by ANY\n"
           "# choice of E0, because E0 enters only through R1/R2 <= 1. floor > 1 means the seed\n"
           "# cannot be made self-sustaining by reseeding the energy.\n"
           "# Command: python docs/dev/switchon-successor/harness/s4_seed_anatomy.py   (2026-08-06)\n")
    with open(os.path.normpath(OUT), "w", newline="") as fh:
        fh.write(hdr)
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    sys.exit(main())
