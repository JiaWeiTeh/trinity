# Phase skeleton — Geometry-set covering-fraction leak (`coverFraction`)

> **Status: SKELETON. No physics is implemented here.** Bodies are `TODO`/`NotImplementedError`.
> This is the S0 deliverable from the leakage spec: a written insertion-point map plus a
> staged plan, grounded in the *current* code. Read this, answer Q1–Q4, get sign-off,
> *then* fill bodies. Line numbers below are anchors and drift — always re-confirm against
> the live file before editing (spec S6).

## 0. Scope & invariants

- **One new input**: `coverFraction` (`Cf`), default `1.0`, validated `0 < Cf ≤ 1`.
- **Hard invariant**: `Cf = 1 ⇒ Lleak ≡ 0, Ṁleak ≡ 0`, byte-for-byte identical trajectory to
  today. This is the merge gate (S4.1). Every stub must short-circuit to zero at `Cf = 1`.
- **Branch note**: spec asks for `feature/leaking-luminosities`; current working branch is
  `claude/dreamy-brown-8G4kF`. Reconcile before merge. Core-energy change ⇒ Joel sign-off.
- **The energy half is small and well-supported. The mass sink (Q2) and X-ray calibration
  (S4.6) are the real work and are deliberately left as flagged stubs below.**

## 1. Insertion-point map (verified against current code)

| # | Site | File · function (anchor) | Live state available | Skeleton action | Gated by |
|---|------|--------------------------|----------------------|-----------------|----------|
| 1 | **Apply leak (explicit)** | `phase1_energy/energy_phase_ODEs.py` · `get_ODE_Edot_pure` (`L_leak = 0`, ~272) | live `R2`, `Eb`, `press_bubble`; frozen `snapshot` | replace `L_leak = 0` with helper call (snapshot `Cf`,`cs` + live `Pb`,`R2`) | C |
| 2 | **Snapshot carries Cf + hot cs** | same file · `ODESnapshot` (~59) & `create_ODE_snapshot` (~133) | — | add `coverFraction`, `c_sound_hot` fields; populate from `params` | A |
| 3 | **Record leak as output** | same file · `compute_derived_quantities`→`ODEResult` (~281, ~398) | live `R2`,`Eb`,`Pb` | add `bubble_Leak` field to `ODEResult`, compute + return it (RHS is *pure*, cannot write) | A |
| 4 | **Apply leak (implicit)** | `phase1b_energy_implicit/get_betadelta.py` (~388, ~483) & `run_energy_implicit_phase.py` (~902) | reads `params` (not pure) | `params['bubble_Leak']` already summed into `L_loss`; **compute it from `Cf`** instead of leaving 0 | C |
| 5 | **Transition double-count guard** | `phase1c_transition/run_transition_phase.py` · ODE (~228–242) | calls `get_ODE_Edot_pure` then `Ed = min(Ed_energy_balance, Ed_soundcrossing)` | decide exclusivity: leak already enters `Ed_energy_balance` via #1 | **Q1**, E |
| 6 | **Mass sink** | `bubble_structure/bubble_luminosity.py` · `mBubble` / `bubble_dMdt` | `Pb`,`T(r)` profile → `n(r)`; mass is *derived*, **not a state** | **STUB** — no ODE state to subtract from; needs Q2 destination | **Q2**, D |
| 7 | **New param + validator** | `_input/registry.py` (`SPECS`, near `bubble_Leak` ~418; validators ~97) | — | add `coverFraction` `ParamSpec` + `_validate_coverFraction`; regen `default.param` | A |
| 8 | **Photon budget `fleak`** | CLOUDY post-processing (out of dynamical ODE) | — | **STUB** — `fleak = 1−Cf` default | **Q4**, G |
| 9 | **X-ray consistency / calibration** | *does not exist in code* (no `L_X`) | bolometric `bubble_LTotal` only | **STUB / separate effort** — calibration target absent | G |

Output plumbing already exists: `bubble_Leak` is registered (`registry.py:418`, default 0),
read back in `_output/trinity_reader.py:238`, and listed in `_input/dictionary.py:1136`.
Today only the *implicit* phase populates it; the energy/transition phases leave it 0.

## 2. Proposed helpers (NOT yet inserted)

Natural home: `trinity/bubble_structure/get_bubbleParams.py` (already holds `bubble_E2P`, `pRam`,
`get_effective_bubble_pressure`). Keep pure: scalars in code units (Msun, pc, Myr) in and out.

```python
# get_bubbleParams.py  (PROPOSED SKELETON)

def get_leak_luminosity(coverFraction, R2, Pb, c_sound_hot, gamma):
    """Enthalpy-flux energy leak, spec Eq. (leak):
        Lleak = gamma/(gamma-1) * (1-Cf) * 4*pi*R2**2 * Pb * c_sound_hot.
    Units: [Msun/pc/Myr^2]*[pc/Myr]*[pc^2] = [Msun*pc^2/Myr^3] (code luminosity) — no conversion.
    c_sound_hot is the HOT-bubble sound speed (get_soundspeed at bubble_Tavg), NOT the shell value.
    Returns 0.0 exactly when Cf == 1.0.
    """
    raise NotImplementedError("Phase C — coefficient is order gamma/(gamma-1); see Q3")


def get_leak_massrate(coverFraction, R2, rho_hot, c_sound_hot):
    """Matching advective mass sink, spec Eq. (mdot):
        Mdotleak = (1-Cf) * 4*pi*R2**2 * rho_hot * c_sound_hot.
    DECISION Q2: where this is debited — no dynamical bubble-mass state exists today
    (Pb is closed as (gamma-1)*Eb/V; mBubble is a derived diagnostic). Returns 0 at Cf == 1.
    """
    raise NotImplementedError("Phase D — destination depends on Q2")
```

## 3. Staged sub-phases (each independently verifiable)

```
A. Plumbing only — no physics change in behaviour
   - add `coverFraction` ParamSpec + `_validate_coverFraction` (registry.py); regen default.param
   - extend ODESnapshot + create_ODE_snapshot with `coverFraction`, `c_sound_hot`
   - add `bubble_Leak` field to ODEResult; populate (=0 while helper is a stub)
   verify: S4.1 regression — rosette param at Cf=1 reproduces R2(t),Eb(t) to solver tol;
           `pytest test/test_engine_purity.py test/test_gen_default_param.py` green.

B. Phase diagnosis — READ-ONLY, decides go/no-go
   - run unmodified param/rosette_sweep*.param; read phase at 2 Myr.
   verify: S4.2 — if energy/transition, leak is the right lever; if already momentum-driven,
           STOP and report (Pb negligible → leak does nothing).

C. Energy leak — apply (explicit RHS #1 + implicit #4)
   - fill get_leak_luminosity; wire site #1 and site #4.
   verify: S4.3 monotonicity — Cf ∈ {1.0, 0.95, 0.9} ⇒ smaller R2,Eb at fixed age;
           bubble_Leak populated & positive.

D. Mass sink — Q2 (HARDEST)
   - fill get_leak_massrate; route per Q2 (evaporation budget vs cooling/X-ray density).
   verify: S4.4 energy+mass audit — full ledger term-by-term; bubble mass falls by ∫Mdotleak dt.

E. Transition exclusivity — Q1
   - guard site #5 so the leak and the sound-crossing drain don't double-count.
   verify: continuity across the transition entry; no discontinuity in Ed.

F. Floor / velocity characterization — decides the manuscript claim
   verify: S4.5 — push Cf to smallest stable value; report min R2 (expect floor near
           momentum-limited radius, > 7 pc) and v_b (expect ≪ 56 km/s).

G. Photon budget (Q4) + X-ray calibration (S4.6) — OUT OF DYNAMICAL ODE
   - fleak = 1−Cf in CLOUDY post-processing; X-ray target absent → separate effort / defer.
```

## 4. Open decisions (blocking — do not silently choose)

- **Q1 — transition double-count.** `run_transition_phase` already calls `get_ODE_Edot_pure`
  (which will contain the leak) then takes `min(Ed_energy_balance, Ed_soundcrossing)`. Either
  zero `Lleak` in the transition phase (its sound-crossing drain *is* the fully-open limit) or
  unify so only one fires. **Concrete, not hypothetical.**
- **Q2 — mass-sink destination.** Forced by code: `Pb = (γ−1)Eb/V`, mass is not a state. Options:
  (a) introduce a tracked bubble-mass/evaporation budget, or (b) feed the density used by
  cooling/X-ray. (a) is a larger change; (b) keeps the diff small but couples to the cooling path.
- **Q3 — energy method.** Faithful covering-fraction leak + mass sink (recommended) vs constant-θ
  radiative proxy (`Lw → (1−θ)Lw`, no mass sink, fast sensitivity check only).
- **Q4 — photon coupling.** `fleak = 1−Cf` tied to the same geometry (recommended) vs independent
  post-processing scalar.

## 5. Verification → real artifacts

| Gate | Artifact in repo |
|------|------------------|
| Regression (S4.1) | `param/rosette_sweep.param`, `rosette_sweep_denser*.param`; `test/test_engine_purity.py` |
| Param schema sync | `tools/gen_default_param.py`, `test/test_gen_default_param.py` |
| Phase diagnosis (S4.2) | run output reader `trinity/_output/trinity_reader.py` (`bubble_Leak`, `bubble_Lloss`) |
| Unit assertion (S2.6) | new test: `Pb·cs·R²` lands in `Msun*pc**2/Myr**3` with no conversion |

## 6. Do-nots (spec S6) + local constraints

- Do **not** add a drag term to the momentum equation — leak enters the **energy** equation only.
- Do **not** freeze `Lleak` per segment — compute it from live `Pb`,`R2` in the RHS. (`cs` may stay
  frozen-per-segment from `bubble_Tavg`; that's the allowed order-unity approximation.)
- Do **not** write to any dict inside `get_ODE_Edot_pure` — it is **pure** (guarded by
  `test_engine_purity.py`). Record `bubble_Leak` via `compute_derived_quantities`/`ODEResult`.
- Do **not** omit the mass sink for the covering-fraction route (Phase D is not optional for the
  X-ray-deficit claim).
- Do **not** call Warpfield's coefficient "wrong": the transition sound-crossing drain is its
  Eq. 20–21, an order-unity multiple of this enthalpy flux; this scheme generalises it.
- Do **not** trust any line number here without re-confirming against the live file.
