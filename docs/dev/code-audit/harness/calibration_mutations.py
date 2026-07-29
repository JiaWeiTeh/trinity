"""Phase 0e: inject known defects, then measure whether the audit finds them.

An audit that finds nothing is indistinguishable from an audit that cannot see.
This seeds `get_InitPhaseParam.py` with eight defects of exactly the classes the
real review is hunting, runs the same blind-lens pipeline over the mutant, and
scores the result. Detection below 6/8 means the prompts are not sensitive
enough and must be strengthened *before* the real run.

Note the design: M1 and M4 mutate the code **and its comment consistently**, so
no amount of reading code-plus-comment can catch them — only Lens C, deriving
the physics independently, can. Those two are the calibration's real test.

    python docs/dev/code-audit/harness/calibration_mutations.py <outdir>
"""

import ast
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[4] / "trinity/phase0_init/get_InitPhaseParam.py"

# (id, defect class, catchable by, old, new)
MUTATIONS = [
    ("M1", "literature coefficient, code+comment consistent", "Lens C only",
     "WEAVER_ENERGY_FRACTION = 5.0 / 11.0", "WEAVER_ENERGY_FRACTION = 5.0 / 7.0"),
    ("M1", "", "", "# Energy fraction in bubble interior: E0 = (5/11) * Lw * dt",
     "# Energy fraction in bubble interior: E0 = (5/7) * Lw * dt"),
    ("M1", "", "", "# From Weaver+77, Eq. 20: E = (5/11) * L_w * t",
     "# From Weaver+77, Eq. 20: E = (5/7) * L_w * t"),
    ("M2", "exponent changed, comment left stale", "A vs B",
     "(Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/35.0)",
     "(Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/25.0)"),
    ("M3", "term dropped from a product", "A vs B/C",
     "         (dt_phase0)**(-6.0/35.0) * \\\n         (1.0 - bubble_xi_Tb)**0.4",
     "         (dt_phase0)**(-6.0/35.0)"),
    ("M4", "geometry factor, code+comment consistent", "Lens C only",
     "dt_phase0 = np.sqrt(3.0 * Mdot0 / (4.0 * np.pi * rhoa * v0**3))",
     "dt_phase0 = np.sqrt(3.0 * Mdot0 / (2.0 * np.pi * rhoa * v0**3))"),
    ("M4", "", "", "# dt = sqrt(3 * Mdot / (4 * pi * rho_a * v^3))",
     "# dt = sqrt(3 * Mdot / (2 * pi * rho_a * v^3))"),
    ("M5", "relation inverted", "A vs B/C",
     "    v0 = 2.0 * Lmech_W / pdot_W", "    v0 = 2.0 * pdot_W / Lmech_W"),
    ("M7", "guard boundary excludes zero", "A vs C",
     "    if nCore <= 0:", "    if nCore < 0:"),
    ("M8", "sign flipped on an exponent", "A vs B",
     "         (dt_phase0)**(-6.0/35.0)", "         (dt_phase0)**(6.0/35.0)"),
]
# Applied last so M2's longer anchor still matches.
M6 = ("M6", "unit conversion dropped", "A dimensional pass",
      "(Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)", "(Lmech_W / WEAVER_L_REF)")


def main(out_dir):
    out = pathlib.Path(out_dir) / "trinity/phase0_init"
    out.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text()
    for mid, _, _, old, new in MUTATIONS:
        assert old in text, f"{mid}: anchor no longer present — the source has moved on"
        text = text.replace(old, new, 1)
    assert M6[3] in text
    text = text.replace(M6[3], M6[4], 1)
    ast.parse(text)
    (out / SRC.name).write_text(text)
    print(f"{len({m[0] for m in MUTATIONS} | {'M6'})} defects -> {out / SRC.name}")


if __name__ == "__main__":
    main(sys.argv[1])
