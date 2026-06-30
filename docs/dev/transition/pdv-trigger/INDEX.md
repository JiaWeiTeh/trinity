# pdv-trigger workstream — master index (START HERE)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a point-in-time map, not a maintained
> spec; the code and sibling docs move. **Re-check each row against the actual file before relying on it.**
>
> 🔄 **Living index — update on every visit.** When you add/rename/retire a doc or finish a task, update the tables
> below (and date it). Keep all banner paragraphs at the top.
>
> 💾 **Persist diagnostics — commit, don't re-run.** Every result has a committed builder + CSV + figure; see
> `REPRODUCE.md` for the result→command→artifact manifest.
>
> 🔗 **Cross-check the sibling docs.** This index is the hub; when a sibling's number/status changes, reconcile it
> here too. Never let two docs disagree.

---

## 0. What this workstream is (one paragraph)

TRINITY transitions a feedback bubble from **energy-driven** to **momentum-driven** when interface cooling drains
the mechanical luminosity (the `cooling_balance` trigger at θ = L_cool/L_mech ≥ 0.95). This workstream asks: *what
sets θ, what knob raises it to the obs/3D values (Lancaster θ~0.9–0.99), and how does it depend on cloud
properties?* It ran from the PdV-in-the-trigger question through the f_κ (Rung-A scalar) calibration to the
conclusion that the faithful fix is the structural **κ_mix (Rung-B) term**, now in its offline-prototype stage.
**Everything to date is dev-only — no production physics code has changed.**

## 1. Read in this order (orientation)

1. **this file** — the map.
2. `PLAN.md` — the living plan + the ⭐ synthesis ("the goal / the merge") + the dated status ledger (newest first).
3. `FINDINGS.md` — the settled, verified results + the taxonomy of approaches.
4. `F_KAPPA_FUNCTIONAL_FORM.md` — the **main current doc**: the f_κ(n) functional form, the 819-sweep scorecard,
   the cliff/fan-out, the metric, the physical derivation → κ_mix (§0–§13).
5. `REPRODUCE.md` — result → `.param`/command → artifact manifest (rebuild any figure without re-running sims).
6. the storyline: `make_pdvtrigger_report.py` → `pdvtrigger_report.html` (rendered narrative, §1–§15).

## 2. The docs — timeline, role, purpose, status

| doc | added | phase / report § | what it is meant to do | status |
|---|---|---|---|---|
| `PLAN.md` | 06-24 | all (the hub) | living plan, ⭐ synthesis, dated status ledger | **live** |
| `NOTE_PATCHES.md` | 06-24 | Phase 1 (trigger) / §2–§3 | the Paper-II note patches: don't-double-count, the f_mix convention fix | settled |
| `FINDINGS.md` | 06-25 | all / §1–§14 | the verified findings + the 3-axis taxonomy (outcome/mechanism/trigger) | **live** |
| `KAPPA_EFF_SCOPING.md` | 06-25 | Phase 1 (mechanism) / §11 | κ_eff Rung-A feasibility map + the back-reaction result (the cooling mechanism) | settled |
| `RUNGB_SCOPING.md` | 06-26 | Phase 2 (Rung B) / §11 | the structural κ_mix scoping; §8 front-conduction next step; §2a θ/λδv reconciliation | **live** (re-promoted) |
| `REPRODUCE.md` | 06-28 | manifest | result→param→command→artifact map; cheap (🟢) vs HPC (🔴) tags | **live** |
| `F_KAPPA_FUNCTIONAL_FORM.md` | 06-29 | Phase 3 (calibration) / §15 | **main doc**: f_κ(n) form, sweep scorecard, cliff, metric, physical-cap, derivation→κ_mix | **live** |
| `KMIX_DIFFUSIVITY.md` | 06-29 | Phase 3 (κ_mix) / §15.7 | the maintainer manuscript draft, verified line-by-line + the λδv-origin refinement | **live** |
| `KMIX_PROTOTYPE.md` | 06-29 | Phase 4 (implementation) | **step 1** of the κ_mix wiring: the offline scoping prototype (units-correct, no solver) | **live** |
| `KMIX_IMPLEMENTATION_SPEC.md` | 06-30 | Phase 4 (implementation) | **design+units spec** for wiring κ_mix: dimensionless-multiplier strategy, the 3 sites, gate param, 8-config gates | **live** (plan; §3 boundary refined by self-consistent) |
| `KMIX_SELFCONSISTENT.md` | 06-30 | Phase 4 (implementation) | **step 2**: κ_mix injected into the REAL solver (monkeypatch). θ rises but **SATURATES** & misses Lancaster for dense clouds; retires the λδv-pin | **live** |
| `PB_COLLAPSE_GUARD_FIX.md` | 06-30 | Phase 4 (hygiene) | plan+tests to stop the energy-collapse reconciliation snapshot emitting a garbage negative Pb | **live** (plan) |

*Phases:* **1** PdV/cooling-boost trigger question (06-24→28) · **2** Rung-B structural scoping (06-26) ·
**3** f_κ calibration + the pivot to κ_mix (06-29, this session) · **4** κ_mix implementation, offline-first (current).

## 3. The live thread — the κ_mix (Rung-B) implementation track

The recurring conclusion is that the faithful fix is `κ = max(κ_mix, κ_Spitzer)`, `κ_mix = (λδv)·n·k_B/μm_p`.
**Hard guardrail (maintainer): no production change before testing all 8 configs, with units handled.** Steps:

| step | what | status | doc |
|---|---|---|---|
| derive | physical prescription → it's κ_mix(λδv), not a scalar power law | ✅ done | `F_KAPPA_FUNCTIONAL_FORM.md` §13 |
| pin λδv | ~~calibrate λδv to Lancaster θ~0.9–0.99~~ | ❌ **RETIRED** — self-consistent θ **saturates** by λδv≈0.01, so λδv is not a knob | `KMIX_SELFCONSISTENT.md` §2 |
| **prototype (offline)** | does κ_mix matter, where, units? — go/no-go | ✅ **GO** — κ_mix dominates the cool layer 10³–10⁸ across nCore 1e2–1e6; full regime set covered **5/5** (4 cal anchors `ok` + heavy `excluded:energy_collapsed`), run in-container 06-30 | `KMIX_PROTOTYPE.md` |
| spec (design) | dimensionless-multiplier κ_eff, gate params, 3 sites, units, 8-config gates | ✅ written (§3 boundary refined) | `KMIX_IMPLEMENTATION_SPEC.md` |
| **self-consistent (offline)** | re-solve structure with κ_mix injected | ✅ **DONE** — G1 bit-identical-off + G2 replay pass; θ RISES but SATURATES & misses Lancaster for dense (1/6 fires); RHS-only stable 6/6; boundary injection diverges | `KMIX_SELFCONSISTENT.md` |
| **strategy revision** | κ_mix is a saturating, density-mismatched correction → combine with θ_target cap? re-metric? boundary re-derive? | ⏳ **maintainer decision** | `KMIX_SELFCONSISTENT.md` §3 |
| gated production | `κ_mix` mode default-off byte-identical; equivalence gate; full 8-config | ⏸ **on hold** pending the strategy revision | `RUNGB_SCOPING.md` §8 + spec §6 |

*Independent hygiene item (not κ_mix):* the energy-collapse reconciliation snapshot emits one garbage negative
Pb on the `fail_repro` heavy run — diagnosed and planned in `PB_COLLAPSE_GUARD_FIX.md` (one-line fix + tests,
queued behind the guardrail).

**Open question carried through:** route (a) diffuse blows out energy-driven (bounded physical diffusivity) vs
route (b) diffuse is 1D-under-cooled → κ_mix gets it to θ_target. The self-consistent run, calibrated to Lancaster,
decides it per cloud. The 8 configs: `simple_cluster`, `midrange_pl0`, `be_sphere`, `pl2_steep`,
`large_diffuse_lowsfe`, `small_dense_highsfe` (6 normal) + `fail_repro` (heavy 5e9) + `small_1e6` (control).

## 4. Data & figures

28 builders + 25 CSVs + 30 figures under `data/` and the folder root — all reproducible without sims. The
canonical map is **`REPRODUCE.md`** (every result → its builder/param/command/artifact). The two HPC artifacts are
the 819-combo sweep (`data/summary.csv`) and its reduction; everything else is a 🟢 read of a committed CSV.

*Index written 2026-06-29 on `feature/PdV-trigger-term-pt2`. Update §2/§3 whenever a doc or step changes.*
