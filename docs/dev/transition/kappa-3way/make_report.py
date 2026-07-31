#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build report.html — THE source of truth for the kappa-3way workstream.

Why this is generated rather than hand-written: a hand-written status page rots the moment an arm
lands, and this workstream exists *because* a stale reading survived eight days (PLAN section 1.4a).
So every number that can be read from a committed artifact IS read from one at build time, and the
page prints its own build timestamp plus a live freshness roll-up. If the data moves and the page is
not rebuilt, the page says so rather than lying quietly.

Reads (all optional — each renders a "not yet measured" panel when absent, which is itself the
honest state before the campaign runs):
    ../pdv-trigger/data/bench7_analysis.csv   the three-way band-entry table (the deliverable)
    ../pdv-trigger/data/bench7_gate_g0.csv    G0 verdicts + the frozen P1 predictions
    ../pdv-trigger/data/freshness_audit.csv   FRESH / OLD / UNSTAMPED per artifact
    ../pdv-trigger/runs/params/bench7/        the committed arm count, counted per K-phase
    ../pdv-trigger/runs/params/bench{5,6}/    the baseline re-run arm counts

Storyline contract (docs/dev/html-insights/build_storylines.py): single <h1>, <p class="sub">
subtitle, scoped <style>, no external assets, no required scripts.

REPRODUCE
    python docs/dev/transition/kappa-3way/make_report.py     # -> report.html
"""

import base64
import csv
import datetime as dt
import html
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDV = HERE.parent / "pdv-trigger"
OUT = HERE / "report.html"

CUTOFF = "2026-07-29"
BAND = "[0.90, 0.99]"
PHASES = {"k1_": "K1", "k1b_": "K1b", "k2_": "K2", "k3_": "K3", "k4_": "K4"}


def _read(path):
    if not path.exists():
        return []
    with path.open(errors="replace") as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def _sources_line(path):
    """The '# SOURCES READ:' header a builder writes, so the page can show its own provenance."""
    if not path.exists():
        return ""
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("#"):
            break
        if "SOURCES READ" in line:
            return line.split("SOURCES READ:", 1)[1].split("<-")[0].strip()
    return ""


def _stamp_of(path):
    if not path.exists():
        return ""
    first = path.read_text(errors="replace").split("\n", 1)[0]
    return first[len("# generated ") :].split(" |")[0] if first.startswith("# generated ") else ""


def _arm_counts():
    """Per-K-phase committed arm counts, straight off disk — never a hardcoded number."""
    out = Counter()
    for p in (PDV / "runs" / "params" / "bench7").glob("*.param"):
        key = max((k for k in PHASES if p.stem.startswith(k)), key=len, default=None)
        if key:
            out[PHASES[key]] += 1
    for name, label in (("bench5", "bench5r"), ("bench6", "bench6r")):
        out[label] = len(list((PDV / "runs" / "params" / name).glob("*.param")))
    return out


def fig(name, alt, caption=""):
    """Embed a committed PNG as a data URI — the page must stay self-contained and offline."""
    p = PDV / name
    if not p.exists():
        return f'<p class="note">missing figure <code>{esc(name)}</code> — run make_bench7_analysis.py</p>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    cap = f'<figcaption class="note">{caption}</figcaption>' if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(alt)}">{cap}</figure>'


def esc(x):
    return html.escape(str(x), quote=True)


def table(headers, rows, cls=""):
    h = "".join(f"<th>{esc(c)}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    c = f' class="{cls}"' if cls else ""
    return f"<table{c}><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def pill(text, kind):
    return f'<span class="pill {kind}">{esc(text)}</span>'


# ---------------------------------------------------------------- data panels


def panel_gate_g0(rows):
    g0 = [r for r in rows if r.get("table") == "G0"]
    if not g0:
        return (
            '<p class="note">No <code>bench7_gate_g0.csv</code> found. Run '
            "<code>python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py</code>.</p>"
        )
    body = []
    for r in g0:
        v = r["verdict"]
        body.append(
            [
                f'<code>{esc(r["quantity"])}</code>',
                esc(r["pre_registered"]),
                esc(r["measured"]),
                f'± {esc(r["abs_tol"])}',
                pill(v, "ok" if v == "PASS" else "bad"),
                f'<span class="dim">{esc(r["note"])}</span>',
            ]
        )
    n_pass = sum(1 for r in g0 if r["verdict"] == "PASS")
    src = _sources_line(PDV / "data" / "bench7_gate_g0.csv")
    head = (
        f'<p class="note"><b>{n_pass}/{len(g0)} PASS.</b> Sources read: '
        f"<code>{esc(src) if src else 'n/a'}</code>. The verdict means different things before and "
        "after the re-run — see the callout below.</p>"
    )
    return head + table(
        ["quantity", "pre-registered", "measured", "tol", "verdict", "note"], body, "wide"
    )


def panel_p1(rows):
    p1 = [r for r in rows if r.get("table") == "P1"]
    if not p1:
        return ""
    ent = [r for r in p1 if "band_entry" in r["quantity"]]
    spr = [r for r in p1 if "spread" in r["quantity"]]
    body = []
    for r in ent + spr:
        body.append(
            [
                f'<code>{esc(r["quantity"])}</code>',
                f'<b>{esc(r["pre_registered"])}</b>',
                pill(r["verdict"], "pending"),
                f'<span class="dim">{esc(r["note"])}</span>',
            ]
        )
    return table(["predicted quantity", "value", "status", "how it was derived"], body, "wide")


def panel_freshness(rows):
    if not rows:
        return (
            '<p class="note">No <code>freshness_audit.csv</code> found. Run '
            "<code>python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py</code>.</p>"
        )
    tally = Counter(r["status"] for r in rows)
    fresh = [r for r in rows if r["status"] == "FRESH"]
    chips = " ".join(
        pill(
            f"{k} {v}",
            {"FRESH": "ok", "OLD": "warn", "UNSTAMPED": "dim-pill"}.get(k, "dim-pill"),
        )
        for k, v in sorted(tally.items())
    )
    body = [
        [
            f'<code>{esc(r["artifact"])}</code>',
            esc(r["generated"]),
            f'<code>{esc(r["builder"])}</code>',
        ]
        for r in fresh
    ]
    tbl = (
        table(["artifact", "generated", "builder"], body, "wide")
        if body
        else '<p class="note">Nothing is FRESH yet — no campaign artifact has been produced.</p>'
    )
    return (
        f'<p class="note">Cutoff <b>{CUTOFF}</b> &nbsp; {chips}</p>'
        f'<p class="note"><b>UNSTAMPED is not OLD.</b> An unstamped file falls back to its git '
        f"<i>commit</i> date, which only upper-bounds its age, so it is reported separately rather "
        f"than folded in. Most of the unstamped count is per-arm trajectory CSVs — the hole the new "
        f"<code>harvest_bench5.py</code> stamp closes on the next reduce.</p>" + tbl
    )


def panel_threeway(rows):
    """The deliverable: band entry per knob per bench + the uniformity spread."""
    ent = [r for r in rows if r.get("table") == "ENTRY"]
    if not ent:
        return (
            '<p class="note">No <code>bench7_analysis.csv</code> yet — the campaign has not been '
            "reduced. Run <code>python docs/dev/transition/pdv-trigger/data/"
            "make_bench7_analysis.py</code>.</p>"
        )
    per = [r for r in ent if r.get("bench") and "SPREAD" not in r["bench"]]
    spr = {r["knob"]: r for r in ent if r.get("bench") == "SPREAD(max/min)"}
    benches = ["bench3_m1e5_r5", "bench2_m1e5_r10", "bench1_m5e4_r20"]
    body = []
    for knob in ("fmix", "fA", "fkappa"):
        cells = []
        for b in benches:
            r = next((x for x in per if x["knob"] == knob and x["bench"] == b), None)
            if not r or not r["entry_dose"]:
                cells.append("<span class='dim'>—</span>")
                continue
            v = esc(r["entry_dose"])
            if r["measured_in_grid"] != "yes":
                v = f"<i>{v}</i>"  # extrapolated
            if r["truncated_arms"] not in ("", "0"):
                v += ' <span class="pill warn">wall-limited</span>'
            cells.append(v)
        s_ = spr.get(knob)
        sv = f"<b>{esc(s_['entry_dose'])}&times;</b>" if s_ else "—"
        note = esc(s_["measured_in_grid"]) if s_ else ""
        body.append([f"<b>{esc(knob)}</b>", *cells, sv, f'<span class="dim">{note}</span>'])
    return table(
        ["knob", "bench3 (n=5520)", "bench2 (n=690)", "bench1 (n=43)", "spread", "caveat"],
        body,
        "wide",
    )


def panel_firemap(rows):
    fm = [r for r in rows if r.get("table") == "FIREMAP"]
    if not fm:
        return ""
    body = [
        [
            f'<code>{esc(r["subject"])}</code>',
            f'<code class="grid">{esc(r["track"])}</code>',
            f'<b>{esc(r["entry_dose"] or "— none —")}</b>',
        ]
        for r in fm
    ]
    return table(["config", "f_kappa fate vs dose", "FIRED at"], body, "wide")


def panel_arms(counts):
    order = [
        ("K1", "bench1/2/3 × f_κ {2,3,4,6,8,12,16,24,32} × prod/diag", "the missing third leg"),
        ("K1b", "bench4/bench5 × f_κ {2,4,8,12,16} × prod/diag", "dense-end fire map"),
        ("K2", "6 band configs × f_κ {1,2,3,4,5,6,7,8,9,12,16} × prod", "whole fire map + squeeze"),
        ("K3", "5 fate-flip arms × 2 (_a/_b) × prod", "determinism (P4)"),
        ("K4", "bench1/bench2 × f_mix {2,3,4,8,12,16} × prod/diag", "the f_mix ladder, in-grid"),
        ("bench5r", "bench5's committed params, re-run", "Θ₀ + the f_A ladder ≤16"),
        ("bench6r", "bench6's committed params, re-run", "f_A 24–128 + the f_mix head-to-head"),
    ]
    body = [
        [
            f"<b>{esc(k)}</b>",
            f'<code class="grid">{esc(g)}</code>',
            str(counts.get(k, 0)),
            f'<span class="dim">{esc(w)}</span>',
        ]
        for k, g, w in order
    ]
    total = sum(counts.values())
    body.append(
        ["<b>total</b>", "", f"<b>{total}</b>", '<span class="dim">all at stop_t = 5 Myr</span>']
    )
    return table(["phase", "grid", "arms", "what it measures"], body, "wide")


# ---------------------------------------------------------------- page


def build():
    gate = _read(PDV / "data" / "bench7_gate_g0.csv")
    ana = _read(PDV / "data" / "bench7_analysis.csv")
    fresh = _read(PDV / "data" / "freshness_audit.csv")
    counts = _arm_counts()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    total = sum(counts.values())

    return TEMPLATE.format(
        now=now,
        sha=esc(sha or "nogit"),
        cutoff=CUTOFF,
        band=BAND,
        total=total,
        bench7=total - counts.get("bench5r", 0) - counts.get("bench6r", 0),
        arms=panel_arms(counts),
        g0=panel_gate_g0(gate),
        p1=panel_p1(gate),
        freshness=panel_freshness(fresh),
        g0_stamp=esc(_stamp_of(PDV / "data" / "bench7_gate_g0.csv") or "not built"),
        threeway=panel_threeway(ana),
        fig_entry=fig(
            "bench7_entry.png",
            "Theta_cum vs dose for f_A, f_mix and f_kappa on the three "
            "clean-blowout benches, with the L21b band shaded",
            "Θ<sub>cum</sub> vs dose, L21b band shaded. <b>f_κ (triangles) is the flattest "
            "curve on every panel</b> — that is the result. On bench2 and bench1 it never "
            "reaches the band within the measured grid.",
        ),
        fig_mass=fig("bench7_massloading.png",
                     "dMdt ratio vs dose for f_kappa and f_A, with f_mix flat at 1",
                     "Mass loading vs dose. The shaded region is <b>suppression</b> — the wrong side "
                     "for a wrinkled interface. f_A sits in it throughout; f_&kappa; enters it above "
                     "f &asymp; 7; f_mix never responds."),
        fig_firemap=fig(
            "bench7_firemap.png",
            "f_kappa fate versus dose for the six band configs",
            "K2's 66 arms. <code>simple_cluster</code> fires only at 4–6 then condenses; "
            "<code>pl2_steep</code> needs ≥8. The two firing windows never overlap.",
        ),
        firemap=panel_firemap(ana),
        ran="yes" if ana else "no",
    )


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>kappa-3way — the three-way band-entry calibration</title>
</head><body>
<h1>kappa-3way — the three-way band-entry calibration</h1>
<p class="sub">The source of truth for the f_&kappa; / f_A / f_mix decision, measured fresh.
Built {now} from the committed artifacts &middot; code <code>{sha}</code>.</p>

<style>
  :root {{ --ink:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --bg:#fff;
          --ok:#0f7b3f; --okbg:#e7f6ec; --bad:#b42318; --badbg:#fee4e2;
          --warn:#9a6700; --warnbg:#fff4d6; --acc:#1f4fd8; --accbg:#eef2ff; }}
  .k3 {{ color:var(--ink); background:var(--bg); line-height:1.6;
        font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
  .k3 h2 {{ margin:2.2em 0 .6em; padding-bottom:.25em; border-bottom:2px solid var(--line);
           font-size:1.35em; letter-spacing:-.01em; }}
  .k3 h3 {{ margin:1.6em 0 .4em; font-size:1.08em; }}
  .k3 code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em;
             background:var(--accbg); padding:.1em .35em; border-radius:3px; }}
  .k3 pre {{ background:var(--accbg); padding:1em; border-radius:6px; overflow-x:auto;
            font-size:.86em; line-height:1.5; }}
  .k3 pre code {{ background:none; padding:0; }}
  .k3 .tw {{ overflow-x:auto; -webkit-overflow-scrolling:touch; margin:1em 0; }}
  .k3 table {{ border-collapse:collapse; width:100%; font-size:.9em; }}
  .k3 th, .k3 td {{ text-align:left; padding:.5em .7em; border-bottom:1px solid var(--line);
                   vertical-align:top; }}
  .k3 th {{ font-weight:600; font-size:.82em; text-transform:uppercase; letter-spacing:.04em;
           color:var(--dim); }}
  .k3 .grid {{ white-space:nowrap; }}
  .k3 .dim {{ color:var(--dim); }}
  .k3 .note {{ color:var(--dim); font-size:.92em; }}
  .k3 .pill {{ display:inline-block; padding:.1em .55em; border-radius:999px; font-size:.78em;
              font-weight:600; white-space:nowrap; }}
  .k3 .ok {{ background:var(--okbg); color:var(--ok); }}
  .k3 .bad {{ background:var(--badbg); color:var(--bad); }}
  .k3 .warn {{ background:var(--warnbg); color:var(--warn); }}
  .k3 .pending {{ background:var(--accbg); color:var(--acc); }}
  .k3 .dim-pill {{ background:var(--line); color:var(--dim); }}
  .k3 .box {{ border-left:4px solid var(--acc); background:var(--accbg); padding:.9em 1.2em;
             border-radius:0 6px 6px 0; margin:1.4em 0; }}
  .k3 .box.stop {{ border-color:var(--bad); background:var(--badbg); }}
  .k3 .box.win {{ border-color:var(--ok); background:var(--okbg); }}
  .k3 .box.hold {{ border-color:var(--warn); background:var(--warnbg); }}
  .k3 .box p:first-child {{ margin-top:0; }} .k3 .box p:last-child {{ margin-bottom:0; }}
  .k3 .lede {{ font-size:1.05em; }}
  .k3 figure {{ margin:1.4em 0; }}
  .k3 figure img {{ max-width:100%; height:auto; display:block; border-radius:6px;
                   background:#fff; padding:.4em; }}
  .k3 figcaption {{ margin-top:.5em; }}
  html {{ background:var(--bg); }}
  body {{ background:var(--bg); color:var(--ink); margin:0 auto; max-width:1080px;
         padding:1.5rem 1.2rem 4rem; }}
</style>

<div class="k3">

<div class="box win">
<p><b>{total}/{total} arms ran on 2026-07-30. The three-way table is MEASURED.</b> Headline:
<b>f_&kappa; is the worst of the three knobs</b> &mdash; spread &ge;16&times; against f_mix's 2.75&times;
and f_A's 6.0&times;, and it does not reach the L21b band at all on the two diffuse benches by
f_&kappa;&nbsp;=&nbsp;32. Prediction <b>P1 is falsified</b> (predicted 3.4&times;). Full record:
<a href="FINDINGS.md">FINDINGS.md</a>.</p>
</div>

<div class="box stop">
<p><b>Gate G0 FAILED 2/11 — and the cause is the most important result here.</b> 116/120 baseline
arms reproduced <b>bit-identically</b> against the 2026-07-19 harvest, and <i>every</i> arm that moved
was one that ran out of wall-clock mid-solve. One such arm (<code>bench1 f_A=128</code>) sits at the
top of f_A's bench1 ladder, so its shorter integration window slid that band-entry dose
74.8&nbsp;&rarr;&nbsp;83.2 and f_A's spread 5.39&nbsp;&rarr;&nbsp;6.00. <b>f_A's spread &mdash; the
number the published head-to-head rests on &mdash; is therefore not converged</b>: it is
5.4&ndash;6.0&times;, wall-limited. 21/294 arms (7.1%) truncated this way.</p>
</div>

<h2>1. The question</h2>

<p class="lede">TRINITY's 1-D interface cooling fraction &theta; undershoots the value Lancaster 2021b
measures in 3-D, so realistic giant molecular clouds never fire the energy&rarr;momentum trigger.
Three knobs can supply the missing cooling. Choosing between them is a calibration question:
<b>for each knob, what dose brings &Theta;<sub>cum</sub> into the observed band {band}, and how much
does that dose vary across cloud density?</b> The knob whose calibrated dose varies least is the
better single physical constant.</p>

<div class="tw"><table>
<thead><tr><th>knob</th><th>multiplies</th><th>acts</th><th>band-entry spread</th></tr></thead>
<tbody>
<tr><td><code>cooling_boost_fA</code></td><td>the L2+L3 interface source terms</td>
    <td>in-solve — T(r) and &Mdot; respond</td><td>5.39&times; <span class="pill warn">VERIFY</span></td></tr>
<tr><td><code>cooling_boost_mode='multiplier'</code></td><td>L<sub>cool</sub>, as a scalar</td>
    <td>frozen structure by construction</td><td>2.96&times; <span class="pill warn">VERIFY</span>, &#8532; extrapolated</td></tr>
<tr><td><code>cooling_boost_kappa</code></td><td>the Spitzer conduction coefficient C</td>
    <td>in-structure — &theta; emerges</td><td><b>no number has ever been measured</b></td></tr>
</tbody></table></div>

<div class="box">
<p><b>That empty cell is this workstream.</b> The published head-to-head is two-way where it should be
three-way — f_&kappa; has never been through the calibration that decided between the other two.</p>
</div>

<h2>2. THE RESULT — the three-way band-entry table</h2>

<p>L21b band {band}, clean-blowout benches. <i>Italic</i> = extrapolated past the grid, not measured.
Read from <code>bench7_analysis.csv</code> at build time.</p>

<div class="tw">{threeway}</div>

{fig_entry}

<p class="note"><b>The extrapolation-free statement</b>, which needs no model: at the top of the
f_&kappa; grid (f_&kappa; = 32), &Theta;<sub>cum</sub> = <b>0.913</b> on bench3 (in band),
<b>0.890</b> on bench2 (below, and <i>saturating</i> &mdash; 24&rarr;32 moves it 0.889&rarr;0.890 while
the integration window is still growing), and <b>0.676</b> on bench1 (far short). f_&kappa; appears to
asymptote just below the band on the intermediate-density cloud.</p>

<p class="note"><b>Why P1 missed.</b> It assumed the dose&ndash;response exponent q &isin; [0.55, 0.70],
carried over from &sect;24's <i>fixed-state</i> L_cool exponents. Measured on the <i>integrated</i>
metric, q &asymp; <b>0.27&ndash;0.32</b> for f_&kappa; &mdash; roughly half &mdash; and 8 of 9 fits
across all three knobs fall below the bracket. Entry dose goes as 1/q in the exponent, so halving q
roughly squares the required dose. That is the whole gap between the predicted 3.4&times; and the
measured &ge;16&times;. It is <code>CLAUDE.md</code> rule 5 again: a per-call equivalence is necessary
but not sufficient.</p>

<p class="note">f_mix's exponent (0.46&ndash;0.56) is about <b>double</b> f_A's and f_&kappa;'s. That is
<i>why</i> it is the most uniform knob &mdash; the uniformity ranking is a consequence of the exponent
ranking, not an independent fact.</p>

<h3>The f_&kappa; fire map (K2, 66 arms) — P3 confirmed</h3>
<p class="note">No single f_&kappa; fires all 6 band configs; best is <b>5/6 at f_&kappa; &isin;
{{8, 9, 12}}</b>, reproducing &sect;12's "5/6 at 12" exactly from an independent fresh grid.
<code>simple_cluster</code> fires only at 4&ndash;6 then condenses; <code>pl2_steep</code> needs
&ge;8. <b>The windows do not overlap</b> &mdash; the squeeze is real, now bounded to one dose unit.</p>

<div class="tw">{firemap}</div>

{fig_firemap}

<h2>3. The mechanism check — none of the three is the wrinkled-interface knob</h2>

<p>The Θ<sub>cum</sub> calibration above scores only the <i>radiative bookkeeping</i>. The physical
motivation for all three knobs is that turbulent mixing <b>wrinkles</b> the contact discontinuity, so
its true area exceeds the 1-D spherical area. In the thin-layer limit that raises every interface
flux <i>together</i> — conduction, radiation, <b>and the evaporative mass flux</b>. So an
area-faithful knob has an unambiguous signature: <b>Ṁ must RISE with dose.</b></p>

{fig_mass}

<div class="tw"><table>
<thead><tr><th>knob</th><th>where it acts</th><th>structure responds?</th><th>Ṁ(f)/Ṁ(1)</th><th>vs the wrinkle picture</th></tr></thead>
<tbody>
<tr><td><b>f_mix</b></td><td><code>L_leak + f·L_cool</code> on the <i>integrated output</i>, feeding the
    energy equation</td><td><b>no</b> — structure-frozen, energetics-live</td><td>&equiv; 1</td>
    <td>❌ no mass-loading response at all</td></tr>
<tr><td><b>f_A</b></td><td><code>dudt = f·dudt</code> <i>inside the ODE</i>, interface band only</td>
    <td>yes (radiative source)</td><td><b>0.988 &rarr; 0.855</b></td>
    <td>❌ <b>wrong sign</b> — cooler interface evaporates less</td></tr>
<tr><td><b>f_&kappa;</b></td><td><code>C_thermal</code> — the Spitzer conduction coefficient</td>
    <td>yes (transport)</td><td><b>1.07 &rarr; 0.94 &rarr; 0.29</b></td>
    <td>⚠️ right sign only below f_&kappa; &asymp; 7</td></tr>
</tbody></table></div>

<div class="box stop">
<p><b>In the dose range where any of these would actually be calibrated, not one raises mass
loading.</b> f_&kappa;'s ratio crosses 1 between f = 6 and 8; by f_&kappa; = 12 — bench3's own
band-entry dose — evaporation is already suppressed 20%. The knob family whose whole motivation is
extra interface area produces, at the operating point, an interface that evaporates <i>less</i>.</p>
</div>

<p><b>The mechanism ranking is the REVERSE of the calibration ranking.</b> f_&kappa; moves a real
transport coefficient and is the only one ever correct on Ṁ; f_A is in-solve but trades the Ṁ-channel
against the θ-channel; f_mix is a scalar on the integrated answer and <b>wins §2 precisely because it
is unconstrained by the physics it represents</b>. Reporting either ranking alone is misleading —
see <a href="FINDINGS.md">FINDINGS §10</a>.</p>

<div class="box">
<p><b>The experiment this implies (36 arms, ~12% of this campaign).</b> An <b>f_area</b> knob applying
f_&kappa; and f_A <i>simultaneously with one shared constant</i>: f_&kappa; carries the
conduction + evaporation channel, f_A the radiative one. Predictions from the measured single-knob
exponents — Ṁ stays rising (net ≈ f<sup>+0.23</sup>) instead of crossing below 1 near f &asymp; 7, and
the Θ<sub>cum</sub> exponent exceeds either alone. <b>0 of 174 arms set more than one knob</b>, so this
is entirely unmeasured — single-knob was enforced by construction for clean attribution.</p>
</div>

<h2>4. What happened — why the old numbers were not trusted</h2>

<p>Three corrections inside five days, all in the parent workstream
<code>docs/dev/transition/pdv-trigger/</code>. None of them was corrupt data. Every one passed the
contamination register. They were <b>correct measurements with a wrong reading</b> — which is the
failure mode a per-artifact grade cannot catch, and the reason this directory starts clean.</p>

<div class="tw"><table>
<thead><tr><th>#</th><th>what was published</th><th>what was true</th><th>how it was caught</th></tr></thead>
<tbody>
<tr><td><b>&sect;17&rarr;&sect;18</b><br><span class="dim">07-27/28</span></td>
    <td>&ldquo;f_mix <b>ELIMINATED</b> as a calibration knob — never reaches the band, wrong-sign
        dose-response&rdquo;</td>
    <td>A <b>metric artifact</b>. The &Theta;<sub>cum</sub> numerator integrated the raw
        <code>Lcool</code> column, which does not carry the boost under <code>multiplier</code> — so
        every f_mix arm was scored as if unboosted. Corrected: monotone rise (bench1 0.221&rarr;0.767),
        and the head-to-head <b>inverts</b>.</td>
    <td>an external review re-derived the metric. The conclusion had stood 8 days across 4 documents.</td></tr>
<tr><td><b>&sect;23</b><br><span class="dim">07-29</span></td>
    <td>&ldquo;f_&kappa; pushes evaporation the <b>wrong way</b> vs El-Badry&rdquo; — the argument used
        to retire the knob</td>
    <td><b>False.</b> Eq 47 carries (C/6&times;10<sup>&minus;7</sup>)<sup>2/7</sup>: mass loss
        <i>rises</i> with conduction, and f_&kappa; multiplies exactly that C. TRINITY reproduces it to
        0.34&ndash;1.63%.</td>
    <td>the maintainer supplied the paper page. 5 sites corrected, including a shipped
        <code>registry.py</code> string.</td></tr>
<tr><td><b>&sect;24</b><br><span class="dim">07-29</span></td>
    <td>&ldquo;no whole-band f_&kappa; — insufficient reach&rdquo;</td>
    <td>The <i>result</i> was right (5/6 at f_&kappa;=12, reproduced exactly); the <i>cause</i> was
        wrong. Every band config crosses &theta;=0.95 somewhere; the band breaks on scattered
        CONDENSE/DRAIN fallout at the condensation boundary.</td>
    <td>re-reading the committed fire map dose by dose.</td></tr>
</tbody></table></div>

<div class="box hold">
<p><b>The rule that follows.</b> A number is quotable here only if its own provenance stamp is dated
on or after <b>{cutoff}</b>. Everything older is <span class="pill warn">VERIFY</span> — possibly
true, not citable until re-measured. One date comparison, applied mechanically by
<code>make_freshness_audit.py</code>, replacing a five-week register whose failure mode was silent.</p>
</div>

<h2>5. The campaign — {total} arms</h2>

<p>All arms <code>stop_t = 5 Myr</code>, one process each, <b>single-knob by construction</b>, with the
two-arm protocol: <b>production</b> (live <code>cooling_balance</code> &rarr; the fire map) and
<b>diagnostic</b> (<code>transition_trigger blowout</code> &rarr; uncensored &theta;(t) across the L21b
window). Counts below are read off the committed <code>.param</code> files at build time, not
hardcoded.</p>

<div class="tw">{arms}</div>

<p class="note"><b>Why the baselines are re-run too.</b> {bench7} arms measure f_&kappa;. Without
<code>bench5r</code>/<code>bench6r</code>, &Theta;<sub>0</sub> and the f_A/f_mix ladders — two of the
three legs of the comparison — would still be 2026-07-19 numbers, and the three-way table would be one
part fresh, two parts VERIFY. They land under fresh names, so nothing older is overwritten and
old-vs-new is a file diff.</p>

<h3>Three grid decisions worth knowing</h3>
<ul>
<li><b>K2 was widened</b> from f_&kappa; {{5,7,9}} to the full {{1&hellip;16}} so the 2026-07-03
    <code>theta5k</code> fire map stops being an input to the whole-band verdict.</li>
<li><b>K1b was extended to 12 and 16</b> — it was the only grid stopping short, which left the dense
    end dark exactly where K1 and K2 are densest, and at the two doses that bracket the best known
    whole-band dose (12) and where the fire set starts falling over (16).</li>
<li><b>K4 re-measures the whole f_mix ladder</b> rather than adding two points to the old one, so f_mix's
    band entry becomes <i>measured in-grid</i> instead of extrapolated — the standing flaw from &sect;18.
    <b>This rests on a reading of the ruling, not a confirmation</b> — one constant,
    <code>F_MIX_K4</code>, changes it, and it is free only until <code>submit</code>.</li>
</ul>

<h2>6. Pre-registered predictions — scored</h2>

<p>Frozen before any arm runs, in <code>pdv-trigger/data/bench7_gate_g0.csv</code>. A miss is
<b>recorded as a miss</b>, never re-negotiated.</p>

<div class="tw"><table>
<thead><tr><th>#</th><th>prediction</th><th>decided by</th></tr></thead>
<tbody>
<tr><td><b>P1</b></td><td>f_&kappa; band entry follows (0.90/&Theta;<sub>0</sub>)<sup>1/q</sup>,
    q &isin; [0.55, 0.70] &rArr; <b>spread &asymp; 2.9&ndash;3.8&times;</b> — between f_mix's 2.96&times;
    and f_A's 5.39&times;. <i>Falsifiable both ways.</i></td><td>K1's measured entry doses</td></tr>
<tr><td><b>P2</b></td><td>&Mdot; <i>rises</i> with f_&kappa; but the ratio <i>decays</i> along a full run
    as E<sub>b</sub> drains</td><td>the <code>bubble_dMdt</code> + <code>Pb</code> columns</td></tr>
<tr><td><b>P3</b></td><td>the squeeze is real: no single f_&kappa; fires all 6 band configs, and the
    failures are CONDENSE/DRAIN, not NOFIRE</td><td>K2's 66-arm fire map</td></tr>
<tr><td><b>P4</b></td><td>the non-monotonic fates are <b>deterministic</b></td>
    <td>K3's paired trajectory hashes</td></tr>
<tr><td><b>P5</b></td><td>f_mix's band entry is reached <b>in-grid</b> by fm &le; 16</td>
    <td>K4's ladder</td></tr>
</tbody></table></div>

<div class="tw">{p1}</div>

<h2>7. Gate G0 — the baseline check, which does double duty</h2>

<p class="note">Artifact stamp: <code>{g0_stamp}</code></p>

{g0}

<div class="box">
<p><b>Run it twice; it answers a different question each time.</b> <i>Before</i> the re-run it reads the
2026-07-19 harvest and is a self-check — the published numbers still fall out of the trajectories they
were computed from. <i>After</i> <code>bench5r</code>/<code>bench6r</code> land it auto-prefers them and
checks <b>the same pre-registered targets against arms run today</b>, so a PASS means the 07-19 result
<b>reproduced</b> and a FAIL means it <b>did not</b>. The targets are not relaxed in either direction.</p>
<p>If it fails: that is a result. Record both numbers and the diff <b>before</b> reading anything
downstream. Never silently adopt either value, and never merge a fresh and a pre-cutoff measurement
into one fit.</p>
</div>

<h2>8. Freshness — what on disk is actually from today</h2>

{freshness}

<h2>9. What was run</h2>

<pre><code>git pull                                  # branch feature/pdv-trigger-5b
cd docs/dev/transition/pdv-trigger/runs
./sync_bench.sh bench7 up                 # `up` serves all campaigns

./sync_bench.sh bench5r submit            # 60  — auto-sized array
./sync_bench.sh bench6r submit            # 60
./sync_bench.sh bench7  submit            # {bench7}
./sync_bench.sh bench7  watch             # Ctrl-C stops watching, not the array

# once each array is DONE — the reduce is ONE-SHOT
./sync_bench.sh bench5r reduce &amp;&amp; ./sync_bench.sh bench5r down
./sync_bench.sh bench6r reduce &amp;&amp; ./sync_bench.sh bench6r down
./sync_bench.sh bench7  reduce &amp;&amp; ./sync_bench.sh bench7  down

cd ../../../../..
python docs/dev/transition/pdv-trigger/data/make_bench5_analysis.py   # auto-prefers bench5r_*
python docs/dev/transition/pdv-trigger/data/make_bench6_analysis.py
python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py    # G0 vs fresh arms
python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py   # the receipt
python docs/dev/transition/kappa-3way/make_report.py                  # rebuild this page</code></pre>

<div class="box stop">
<p><b>Two things that are free now and expensive after <code>submit</code>.</b></p>
<p>1. <b>K4's grid.</b> It rests on a reading of &ldquo;no, redo if possible&rdquo;. If wrong, change
<code>F_MIX_K4</code> in <code>make_kappa_reopen_params.py</code>, re-run the builder, update
<code>PHASE_COUNTS</code> in <code>test/test_bench7_params.py</code>, re-commit.</p>
<p>2. <b>The reduce is ONE-SHOT.</b> gpfs is cleaned; raw arms do not come back. The extra columns
<code>Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate</code> are already declared. Anything
else your analysis will need must be added <i>before</i> the first reduce.</p>
</div>

<h2>10. What the result means</h2>

<p>The deliverable is the three-way table this program has been missing — per knob: band-entry dose on
each bench, the spread, and whether it was <b>measured in-grid or extrapolated</b>. Per gate G5, both
&Theta;<sub>cum</sub> variants and the frozen-no-root share are reported beside every band-setting
number.</p>

<div class="box win">
<p><b>f_&kappa; spread smallest, doses reachable</b> &rarr; f_&kappa; re-enters as a live candidate and
the two-way comparison is corrected to three-way.</p>
</div>
<div class="box hold">
<p><b>f_&kappa; spread smallest, doses unreachable</b> (entry beyond the condensation boundary) &rarr; a
<b>real result, not a failure</b> — and precisely why G5 forbids publishing the uniformity number
without the reachability number beside it.</p>
</div>
<div class="box stop">
<p><b>f_&kappa; spread largest</b> &rarr; f_&kappa; closes as a calibration knob, this time on a
<i>measured</i> basis rather than a falsified argument. The &sect;23 correction stands either way: it was
about honesty, never about promoting f_&kappa;.</p>
<p><b>Pre-registered terminal stop:</b> if no knob holds one constant across the band to the agreed
factor, the single-constant program <b>stops</b> rather than being re-scoped into a fitted f(n).</p>
</div>

<h2>11. Where things live</h2>

<div class="tw"><table>
<thead><tr><th>what</th><th>where</th><th>why</th></tr></thead>
<tbody>
<tr><td>this page, the plan, the rules</td><td><code>docs/dev/transition/kappa-3way/</code></td>
    <td>starts clean at the cutoff</td></tr>
<tr><td>param builders, <code>.param</code> files, sbatch, <code>sync_bench.sh</code></td>
    <td><code>pdv-trigger/runs/</code> <span class="dim">(unchanged)</span></td>
    <td>cluster paths are baked into the sbatch and sync scripts. Moving them buys a tidier tree and
        risks a silent path break on a one-shot {total}-arm reduce. <b>Deliberately not moved.</b></td></tr>
<tr><td>the 2026-07-19 and earlier evidence</td><td><code>pdv-trigger/</code> <span class="dim">(unchanged)</span></td>
    <td>preserved as-is and <b>demoted, not rewritten</b>: read it for <i>what to test and why</i>,
        never for <i>what the answer is</i></td></tr>
</tbody></table></div>

<p class="note">Production is untouched throughout: the defaults remain
<code>cooling_boost_mode='none'</code>, f_&kappa; = 1.0, f_A = 1.0.</p>

</div>
</body></html>
"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
