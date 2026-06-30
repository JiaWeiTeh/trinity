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
sets θ, what raises it to the obs/3D values (Lancaster θ~0.9–0.99), and how does it depend on cloud properties?*
**Current direction (2026-06-30):** the one master parameter is θ ≡ L_cool/L_mech — *identical* in TRINITY,
El-Badry, and Lancaster — and El-Badry gives a 3D-calibrated **closed form θ(λδv, n)**. The plan is to **impose
that θ as the trigger target** via TRINITY's gated `theta_target` mode (λδv≈3, n=local cloud density, θ_max
ceiling, paired with `ebpeak` for PdV). *History:* it ran from the PdV question → f_κ (Rung-A) calibration → a
structural κ_mix (Rung-B) port that was **tested and shelved** (it saturates) → the θ_target direction.
**Everything to date is dev-only — no production physics code has changed** (except the queued Pb-collapse hygiene
fix, applied + tested). See `PLAN.md` ⭐⭐ canonical synthesis + §1.5 staleness audit above.

## 1. Read in this order (orientation) — updated 2026-06-30 for the θ_target direction

1. **this file** — the map (incl. the §1.5 staleness audit below).
2. `PLAN.md` → the **⭐⭐ CANONICAL SYNTHESIS + VERDICT** block (the current direction; supersedes all earlier
   synthesis) + the dated status ledger (newest first).
3. `ELBADRY_REFERENCE.md` + `LANCASTER_REFERENCE.md` — 📌 the two **imprint** reference docs (θ definition, the
   closed form, λδv≈3, the n-mapping, the theta_target verification, PdV). Read these instead of the PDFs.
4. `KMIX_SELFCONSISTENT.md` — *why the structural κ_mix port was shelved* (the negative result that pivoted us).
5. `REPRODUCE.md` — result → `.param`/command → artifact manifest.

> ⚠️ **`F_KAPPA_FUNCTIONAL_FORM.md` is NO LONGER the main doc** — it documents the *f_κ(n) power-law* avenue,
> which is **superseded** (see §1.5). Read it as history, not direction.

## 1.5 ⚠️ STALENESS AUDIT — docs that describe SUPERSEDED directions (read before trusting a conclusion)

The direction changed on 2026-06-30 (κ_eff/f_κ/κ_mix-structural → **impose El-Badry's θ as the trigger
target**). Several docs predate that and, read in isolation, would point the wrong way. **They are kept for
provenance but flagged here so a stale conclusion can't hijack the path forward:**

| doc | what's STALE in it | the correct current view |
|---|---|---|
| `F_KAPPA_FUNCTIONAL_FORM.md` | the whole **f_κ(n) power-law** program (f_κ∝n^−0.3/−0.6, the 819-sweep scorecard, the "cliff") as the *direction* | f_κ is a tunable-but-unphysical fudge; **superseded** by imposing El-Badry's θ_target. The sweep data is still valid *evidence*; the **prescription is not the plan**. |
| `RUNGB_SCOPING.md` | the **structural κ_mix injection** ("re-promoted", §8 gated production) as the path | the structural port is **SHELVED** (saturates/unstable, `KMIX_SELFCONSISTENT.md`); κ_mix survives only as physical *justification* for θ∝√(λδv·n) |
| `KMIX_SELFCONSISTENT.md` §2 | "dense θ plateaus low (~0.35) / only 1/6 fires" | **WALKED BACK** — that was the wrong epoch (blowout) + a buggy port; El-Badry+Lancaster agree **dense θ is HIGH (0.9–0.99)**. See §2b and `LANCASTER_REFERENCE.md` §7. |
| `KMIX_DIFFUSIVITY.md` / `KMIX_PROTOTYPE.md` | "calibrate λδv to Lancaster (value open)"; prototype Pb anchors from **0.3–1.0 Myr truncated** runs | **λδv≈3 is now pinned** (`LANCASTER_REFERENCE.md` §7); re-derive prototype Pb from ≥5 Myr runs before quoting numbers |
| `KMIX_IMPLEMENTATION_SPEC.md` | the κ_mix-into-the-ODE wiring design | **SHELVED** (banner in the doc); its dimensionless-multiplier *units* strategy is still reusable |
| any "Lancaster 2021c / ApJ 914, 91" / "ApJ 914,90 = theory" | paper-ID confusion | ApJ 914, **90 is Paper II (sims)** — the θ~0.9–0.99 anchor; see `LANCASTER_REFERENCE.md` §0 |

**Rule going forward (maintainer): whenever a decision is made, update the ⭐⭐ canonical synthesis AND this
audit AND the affected sibling together — never one in isolation.**

## 2. The docs — timeline, role, purpose, status

| doc | added | phase / report § | what it is meant to do | status |
|---|---|---|---|---|
| `PLAN.md` | 06-24 | all (the hub) | living plan, ⭐ synthesis, dated status ledger | **live** |
| `NOTE_PATCHES.md` | 06-24 | Phase 1 (trigger) / §2–§3 | the Paper-II note patches: don't-double-count, the f_mix convention fix | settled |
| `FINDINGS.md` | 06-25 | all / §1–§14 | the verified findings + the 3-axis taxonomy (outcome/mechanism/trigger) | **live** |
| `KAPPA_EFF_SCOPING.md` | 06-25 | Phase 1 (mechanism) / §11 | κ_eff Rung-A feasibility map + the back-reaction result (the cooling mechanism) | settled |
| `RUNGB_SCOPING.md` | 06-26 | Phase 2 (Rung B) / §11 | the structural κ_mix scoping; §8 front-conduction next step; §2a θ/λδv reconciliation | 🛑 **SHELVED** (structural port abandoned; §1.5) |
| `REPRODUCE.md` | 06-28 | manifest | result→param→command→artifact map; cheap (🟢) vs HPC (🔴) tags | **live** |
| `F_KAPPA_FUNCTIONAL_FORM.md` | 06-29 | Phase 3 (calibration) / §15 | f_κ(n) form, sweep scorecard, cliff, metric, derivation→κ_mix | 🛑 **SUPERSEDED direction** (data valid; prescription not the plan; §1.5) |
| `KMIX_DIFFUSIVITY.md` | 06-29 | Phase 3 (κ_mix) / §15.7 | the maintainer manuscript draft, verified line-by-line + the λδv-origin refinement | **live** |
| `KMIX_PROTOTYPE.md` | 06-29 | Phase 4 (implementation) | **step 1** of the κ_mix wiring: the offline scoping prototype (units-correct, no solver) | **live** |
| `KMIX_IMPLEMENTATION_SPEC.md` | 06-30 | Phase 4 (implementation) | **design+units spec** for wiring κ_mix: dimensionless-multiplier strategy, the 3 sites, gate param, 8-config gates | **live** (plan; §3 boundary refined by self-consistent) |
| `KMIX_SELFCONSISTENT.md` | 06-30 | Phase 4 (implementation) | **step 2**: κ_mix in the REAL solver (monkeypatch). θ SATURATES (retires λδv-pin); **§2b time-resolved**: blowout was the wrong epoch — mid clouds would fire, dense stay low, early phase needs a smooth-max injection | **live** (dense-low walked back, see ELBADRY_REFERENCE) |
| `ELBADRY_REFERENCE.md` | 06-30 | Phase 4 (the pivot) | 📌 **full El-Badry+2019 distilled** (every eq/number; skip the PDF). θ_ElBadry = θ_TRINITY; closed form Eq 37/38; TRINITY mapping + theta_target verification | **live** (imprint) |
| `LANCASTER_REFERENCE.md` | 06-30 | Phase 4 (the pivot) | 📌 **Lancaster distilled** (2025 CEM PDF + 2021 Paper II sims; skip the PDFs). θ=Ė_cool/Lw matches; the αp (PdV/momentum) split; θ~0.9–0.99 + λδv≈3 + route-a (§7) | **live** (imprint) |
| `THETA_ELBADRY_SPEC.md` | 06-30 | Phase 5 (implementation) | **the capstone spec**: gated `theta_elbadry` mode (3 params + 1 `effective_Lloss` branch), θ_max, ebpeak pairing, byte-identical-off, 8-config ≥5 Myr test | **live** (implementation-ready) |
| `PB_COLLAPSE_GUARD_FIX.md` | 06-30 | Phase 4 (hygiene) | the energy-collapse reconciliation no longer emits a garbage negative Pb — **APPLIED + tested** (596 pass) | **done** |

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
| **self-consistent (offline)** | re-solve structure with κ_mix injected | ✅ **DONE** — G1 bit-identical-off + G2 replay pass; θ RISES but SATURATES (λδv not a dial); RHS-only stable; boundary injection diverges | `KMIX_SELFCONSISTENT.md` §2 |
| **time-resolved θ (re-metric)** | is the dense ceiling real or a single-row artifact? | ✅ **DONE** — blowout was the WRONG epoch; θ peaks early; mid (1e4) would fire, dense (≥1e5) stay low; **but early high-Pb epochs don't solve** (hard-max too stiff) + kprime bug (−1/T) | `KMIX_SELFCONSISTENT.md` §2b |
| ~~revise injection~~ | ~~smooth-max + correct kprime~~ | ⏸ **SHELVED** — superseded by the El-Badry θ_target path (no κ_mix port needed) | `ELBADRY_REFERENCE.md` |
| **El-Badry θ_target (current direction)** | calculator + mechanism verified + n-mapping resolved + λδv≈3 calibrated + spec written (two-stage) | ⏳ **NEXT = STAGE A SHADOW** — run the θ_elbadry logic via monkeypatch on 8 configs ≥5 Myr (no `trinity/` edit), resolve max-vs-direct + fate from data; **Stage B production only after** | `THETA_ELBADRY_SPEC.md` §8, `ELBADRY_REFERENCE.md` §7–§9, `LANCASTER_REFERENCE.md` §7 |
| gated production | `κ_mix` mode default-off byte-identical; equivalence gate; full 8-config | ⏸ **on hold** pending the strategy revision | `RUNGB_SCOPING.md` §8 + spec §6 |

*Independent hygiene item (not κ_mix), ✅ APPLIED 2026-06-30:* the energy-collapse reconciliation used to emit
one garbage negative Pb on collapsed runs (e.g. `fail_repro`). Fixed in `run_energy_implicit_phase.py` (skip the
Pb recompute on the `energy_collapsed` exit, but still `save_snapshot()` so code 51 propagates) + a failing-first
test; full suite 596 pass. Details + the trinity-not-bit-reproducible finding: `PB_COLLAPSE_GUARD_FIX.md`.

**Open question carried through:** route (a) diffuse blows out energy-driven (bounded physical diffusivity) vs
route (b) diffuse is 1D-under-cooled → κ_mix gets it to θ_target. The self-consistent run, calibrated to Lancaster,
decides it per cloud. The 8 configs: `simple_cluster`, `midrange_pl0`, `be_sphere`, `pl2_steep`,
`large_diffuse_lowsfe`, `small_dense_highsfe` (6 normal) + `fail_repro` (heavy 5e9) + `small_1e6` (control).

## 4. Data & figures

28 builders + 25 CSVs + 30 figures under `data/` and the folder root — all reproducible without sims. The
canonical map is **`REPRODUCE.md`** (every result → its builder/param/command/artifact). The two HPC artifacts are
the 819-combo sweep (`data/summary.csv`) and its reduction; everything else is a 🟢 read of a committed CSV.

*Index written 2026-06-29 on `feature/PdV-trigger-term-pt2`. Update §2/§3 whenever a doc or step changes.*
