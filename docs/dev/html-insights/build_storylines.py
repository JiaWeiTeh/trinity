#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_storylines.py — compose the per-storyline "book" HTML files under
docs/dev/html-insights/.

Each TRINITY dev investigation already renders a self-contained HTML report (its
generator lives with its workstream and base64-embeds its own figures). This
script *merges* those reports into a small number of chaptered "storyline books"
so a reader can follow one narrative arc end to end. A book is itself fully
self-contained: one MathJax include + a sticky chapter TOC + the three dev-doc
banners, then the chapters.

Merge mechanics (per chapter, content-agnostic so the arc can be re-wired freely):
  * pull the body content out of the source report (its <div class="wrap"> inner
    HTML, or the raw <body> for the plain-body reports);
  * scope that report's own <style> under a per-chapter id, so chapters that
    happen to define the same class (.sub, .tag, .num, ...) cannot collide;
  * drop the report's <h1> (the book assigns the chapter title) and demote the
    remaining headings one level (report h2 -> book h3, h3 -> h4);
  * namespace element ids + in-page #anchors with the chapter slug.
The figures ride along untouched because they are inline base64 data URIs.

SOURCES ARE THE SOURCE OF TRUTH. This script does not invent prose; it composes
already-rendered reports. To fix a claim, fix that report's generator, re-run it,
then re-run this. The one exception is an "inline" chapter (a short authored
bridge, e.g. a plan with no HTML report) whose HTML lives in the registry below.

REPRODUCE:
    cd /home/user/trinity
    python docs/dev/html-insights/build_storylines.py
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent  # docs/dev/html-insights
REPO = HERE.parents[2]  # repo root
DEV = HERE.parent  # docs/dev

# --- authored "bridge" chapters (no standalone report; verified 2026-06-22) --- #
COOLING_BRIDGE_HTML = """
<p class="chlede">A planned decoupling of the cooling-table loaders from their hardcoded
SB99 / OPIATE / CLOUDY assumptions — recorded here so the table audits read as a program,
not a one-off.</p>
<h2>What the audit proposes</h2>
<p>The non-CIE cooling loader hardcodes the SB99 coupling and the column names
(<code>ndens/temp/phi/cool/heat</code>); the CIE path selects tables by integer index with a
silent fall-through for unrecognised metallicities; an unused <code>metallicity</code> argument
lingers on <code>CIE.get_Lambda</code>. <code>docs/dev/cooling/refactor-audit.md</code> lays out
PR-1&ndash;PR-4 to decouple these.</p>
<h2>What has shipped</h2>
<ul>
<li>The cooling-table <b>temperature floor</b> fix (Chapter 1) &mdash;
<code>net_coolingcurve.py:130&ndash;131</code>, commit <code>cc8ae76</code>.</li>
<li><b><code>NameError &rarr; ValueError</code></b> for an unsupported <code>ZCloud</code> &mdash;
<code>read_cloudy.py:298&ndash;302</code>, commit <code>3deec3d</code>.</li>
</ul>
<h2>Still pending</h2>
<ul>
<li>SB99 coupling still hardcoded in <code>read_cloudy.py:48&ndash;49</code> and
<code>get_filename()</code> (<code>:267&ndash;340</code>).</li>
<li>Hardcoded column names at <code>read_cloudy.py:182&ndash;187</code>.</li>
<li>CIE path still integer-index with silent fall-through (<code>read_param.py:417&ndash;429</code>).</li>
<li>Unused <code>metallicity</code> arg on <code>CIE.get_Lambda</code> (<code>read_coolingcurve.py:25</code>).</li>
</ul>
<p>Status: <b>actionable</b> &mdash; core refactor unbuilt. Line refs in
<code>refactor-audit.md</code> have drifted 5&ndash;45 lines; the current positions are above.</p>
"""

PSHADOW_EPILOGUE_HTML = """
<p class="chlede">A path we explored and dropped &mdash; recorded so the arc shows why the
trigger question landed where it did.</p>
<p>Between the speed-ups and the clean-room investigation, an intermediate design
(<code>transition/pshadow-design.md</code>, with the <code>P0.md</code> harvest) proposed a
two-criterion trigger &mdash; <b>F0 cooling &or; F4 blowout</b> &mdash; on the premise that
<i>flat</i> clouds transition by cooling (F0 fires) while <i>steep</i> ones blow out. The
clean-room redo (Chapter&nbsp;2) <b>falsified the cooling half</b>: across six regime-spanning
configs F0 never fires &mdash; even a flat <code>simple_cluster</code> blows out geometrically
while its cooling ratio floors at 0.40. So pshadow / P0 are <b>superseded</b>, not vindicated,
and nothing from them shipped (<code>git grep transition_trigger|blowout|shadow trinity/</code>
&rarr; empty). The live question is the geometric / E<sub>b</sub>-peak handoff and the missing
mixing-layer cooling, not a two-criterion cooling test.</p>
"""

# --------------------------------------------------------------------------- #
# Chapter source registry. Each storyline -> ordered chapters. A chapter is
# either {"type": "html", "src": <rendered report>} or
# {"type": "inline", "html": "<authored fragment>"}.
# Refined as verification ledgers land (docs/dev/html-insights/verification/).
# --------------------------------------------------------------------------- #
STORYLINES = [
    {
        "slug": "s0",
        "title": "The β–δ implicit solver & “Problem 2”",
        "subtitle": "Repairing the clamped β–δ energy-phase solver — the root-finding "
        "maps, the cage diagnostic, and the velocity-arms hunt that set up the "
        "transition-trigger investigation. Kept as its own book so s1 stays light.",
        "chapters": [
            {
                "title": "The β–δ implicit solver & “Problem 2”",
                "type": "html",
                "src": DEV / "archive/betadelta/insights_betadelta_illustrated.html",
            },
        ],
    },
    {
        "slug": "s1",
        "title": "From the β–δ solver to the transition-trigger problem",
        "subtitle": "Making the implicit energy-phase solver fast, then discovering "
        "it can never leave the energy phase.",
        "chapters": [
            {
                "title": "Speed-ups: the bubble-luminosity performance story",
                "type": "html",
                "src": DEV / "performance/F1_REPORT.html",
            },
            {
                "title": "The transition-trigger problem — geometric, not thermal",
                "type": "html",
                "src": DEV / "transition/cleanroom/transition_report.html",
            },
            {
                "title": "Re-examining the verdict — bug, fake boundary, or a real exit? (part 4)",
                "type": "html",
                "src": DEV / "transition/pt4/pt4_transition_report.html",
            },
            {
                "title": "Postscript — the superseded pshadow / P0 proposal",
                "type": "inline",
                "unnumbered": True,
                "html": PSHADOW_EPILOGUE_HTML,
            },
        ],
    },
    {
        "slug": "s2",
        "title": "The ODE-solver saga — LSODA → solve_ivp",
        "subtitle": "An LSODA warning flood in the shell ODE, the replay harness "
        "that diagnosed it, and what actually fixed it.",
        "chapters": [
            {
                "title": "The shell ODE: odeint, the LSODA flood, and the fix",
                "type": "html",
                "src": DEV / "shell-solver/insights.html",
            },
        ],
    },
    {
        "slug": "s3",
        "title": "Why large clouds failed (helix)",
        "subtitle": "A regime-spanning crash in the energy phase for massive "
        "clouds — the mechanism, why only the massive ones, and the fix.",
        "chapters": [
            {
                "title": "Failed large clouds — diagnosis & fix",
                "type": "html",
                "src": DEV / "failed-large-clouds/insights.html",
            },
            {
                "title": "Heavy clouds, part 4 — can we keep them alive?",
                "type": "html",
                "src": DEV / "transition/pt4/pt4_heavy_report.html",
            },
        ],
    },
    {
        "slug": "s4",
        "title": "Hidden constants & table audits",
        "subtitle": "Magic numbers buried in the cooling tables — found, measured, "
        "and corrected.",
        "chapters": [
            {
                "title": "The cooling-table temperature floor",
                "type": "html",
                "src": DEV / "magic-numbers/tclamp_report.html",
            },
            {
                "title": "The cooling-table refactor (planned)",
                "type": "inline",
                "html": COOLING_BRIDGE_HTML,
            },
        ],
    },
]


# --------------------------------------------------------------------------- #
# Extraction + transforms
# --------------------------------------------------------------------------- #
def _strip_scripts(html: str) -> str:
    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)


def extract_content(raw: str) -> str:
    """Inner HTML of the report body, with one outer .wrap div unwrapped."""
    b0 = re.search(r"<body[^>]*>", raw, re.I)
    b1 = raw.lower().rfind("</body>")
    inner = raw[b0.end() : b1] if b0 else raw
    inner = _strip_scripts(inner)
    m = re.match(r"\s*<div\s+class=\"wrap\"[^>]*>", inner, re.I)
    if m:
        inner = inner[m.end() :]
        k = inner.rfind("</div>")
        if k != -1:
            inner = inner[:k]
    return inner.strip()


def _drop_heading_rules(css: str) -> str:
    """Drop rules whose selectors are ALL bare h1..h6 — headings are governed
    uniformly by BOOK_CSS; source rules are keyed to pre-demotion tags."""

    def repl(m):
        sels = [s.strip() for s in m.group(1).split(",") if s.strip()]
        if sels and all(re.fullmatch(r"h[1-6]", s, re.I) for s in sels):
            return ""
        return m.group(0)

    return re.sub(r"([^{}]+)\{[^{}]*\}", repl, css)


def extract_style(raw: str) -> str:
    css = "\n".join(m.group(1) for m in re.finditer(r"<style[^>]*>(.*?)</style>", raw, re.S | re.I))
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)  # drop comments
    return _drop_heading_rules(css)


def _scope_flat(css: str, scope: str) -> str:
    """Prefix each selector of every brace-free rule with `scope`."""

    def repl(m: re.Match) -> str:
        sels, body = m.group(1), m.group(2)
        if sels.strip().startswith("@"):  # @font-face / @keyframes body
            return m.group(0)
        new = []
        for s in sels.split(","):
            s = s.strip()
            if not s:
                continue
            new.append(scope if s in (":root", "html", "body", "*") else f"{scope} {s}")
        return f"{', '.join(new)}{{{body}}}"

    return re.sub(r"([^{}]+)\{([^{}]*)\}", repl, css)


def scope_css(css: str, scope: str) -> str:
    """Scope a stylesheet under `scope`, handling @media (scope inner rules)."""
    out, pos = [], 0
    for m in re.finditer(r"@media[^{]*\{", css):
        out.append(_scope_flat(css[pos : m.start()], scope))
        depth, j = 1, m.end()
        while j < len(css) and depth:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        out.append(css[m.start() : m.end()])  # "@media ... {"
        out.append(_scope_flat(css[m.end() : j - 1], scope))
        out.append("}")
        pos = j
    out.append(_scope_flat(css[pos:], scope))
    return "".join(out)


def _slug(text: str) -> str:
    t = re.sub(r"<[^>]+>", "", text)
    t = re.sub(r"&[a-z]+;", " ", t)
    t = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return t[:48] or "sec"


def ensure_h2_ids(content: str) -> str:
    """Give every report <h2> an id (from its text) if it lacks one."""

    def repl(m: re.Match) -> str:
        attrs, inner = m.group(1), m.group(2)
        if re.search(r"\bid=", attrs):
            return m.group(0)
        return f'<h2{attrs} id="{_slug(inner)}">{inner}</h2>'

    return re.sub(r"<h2([^>]*)>(.*?)</h2>", repl, content, flags=re.S | re.I)


def demote_headings(content: str, by: int = 1) -> str:
    for n in range(6, 0, -1):
        tgt = min(n + by, 6)
        content = re.sub(rf"(</?\s*)h{n}(\b)", rf"\g<1>h{tgt}\g<2>", content, flags=re.I)
    return content


def namespace(content: str, prefix: str) -> str:
    content = re.sub(r"\bid=\"([^\"]+)\"", lambda m: f'id="{prefix}-{m.group(1)}"', content)
    content = re.sub(r"href=\"#([^\"]+)\"", lambda m: f'href="#{prefix}-{m.group(1)}"', content)
    return content


def drop_first_h1(content: str) -> tuple[str, str]:
    """Remove the report's <h1> (and an immediately-following .sub) ; return
    (content_without_h1, subtitle_html_or_empty)."""
    sub = ""
    sm = re.search(r"<(p|div)\s+class=\"sub\"[^>]*>.*?</\1>", content, flags=re.S | re.I)
    content = re.sub(r"<h1[^>]*>.*?</h1>", "", content, count=1, flags=re.S | re.I)
    if sm and sm.start() < 400:
        sub = sm.group(0)
        content = content.replace(sub, "", 1)
    return content.strip(), sub


# --------------------------------------------------------------------------- #
# Book assembly
# --------------------------------------------------------------------------- #
def build_chapter(
    slug: str, idx: int, ch: dict, num: int | None
) -> tuple[str, list[tuple[str, str]]]:
    cid = f"{slug}-ch{idx}"
    if ch["type"] == "inline":
        body = namespace(demote_headings(ensure_h2_ids(ch["html"])), cid)
        style = ""
        sub = ""
    else:
        raw = Path(ch["src"]).read_text(encoding="utf-8", errors="replace")
        scoped = scope_css(extract_style(raw), f"#{cid}-body")
        # Overflow guard: cap EVERY image (and code block) to the column width — the
        # reports' own CSS often only targets `figure.fig img`, so `.grid2` cells, bare
        # <img>, and embedded GIFs would otherwise render at natural size and overflow.
        guard = (
            f"#{cid}-body img{{max-width:100%;height:auto}}"
            f" #{cid}-body pre{{max-width:100%;overflow-x:auto}}"
        )
        style = f"<style>{scoped}\n{guard}</style>\n"
        content = ensure_h2_ids(extract_content(raw))
        content, sub = drop_first_h1(content)
        body = namespace(demote_headings(content), cid)
        sub = namespace(demote_headings(sub), cid) if sub else ""
    # sub-TOC: the report's (now demoted) h3 sections with ids
    subsecs = [
        (m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip())
        for m in re.finditer(r"<h3[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</h3>", body, flags=re.S | re.I)
    ]
    if num is None:
        head = f"<h2 class=\"ch-title\" id=\"{cid}\">{ch['title']}</h2>\n"
    else:
        head = (
            f'<h2 class="ch-title" id="{cid}">'
            f"<span class=\"ch-num\">Chapter {num}</span>{ch['title']}</h2>\n"
        )
    section = f'{style}<section class="chapter" id="{cid}-body">\n{head}{sub}\n{body}\n</section>'
    return section, [(cid, ch["title"])] + [(a, t) for a, t in subsecs]


BANNERS = """
<div class="banners">
  <p class="bn warn"><b>⚠️ This document may be out of date — verify before trusting it.</b>
  It is a point-in-time analysis/audit, not a maintained spec; the code moves faster than
  these notes (paths, line numbers, and “what shipped” status drift). Any agent or
  person reading this: treat it as unverified — re-check each claim, snippet, and line
  reference against the current source before relying on it.</p>
  <p class="bn live"><b>🔄 Living book — recheck and refine on every visit.</b> This is an
  evolving narrative, not a frozen record. Whoever opens it should re-verify the claims and
  line references against current source, update anything that has drifted, and rethink the
  arc itself if a better ordering or framing exists — leave it better than you found it.</p>
  <p class="bn keep"><b>💾 Persist diagnostics — commit, don't re-run.</b> The container is
  ephemeral and the full/hybr runs cost hours; every figure and number here traces to a
  committed harness/CSV in the relevant <code>docs/dev/&lt;workstream&gt;/</code> folder, so a
  future visit reproduces or compares <em>without</em> re-running.</p>
</div>
"""

BOOK_CSS = """
:root{--ink:#1f2733;--mut:#5b6675;--line:#e3e8ef;--bg:#ffffff;--panel:#f7f9fc;--accent:#2c6fb3;}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.book{max-width:880px;margin:0 auto;padding:38px 22px 110px;}
.book>h1{font-size:31px;line-height:1.2;margin:0 0 6px;letter-spacing:-.01em;}
.book>.lede{color:var(--mut);font-size:16px;margin:0 0 26px;}
.banners{margin:22px 0 8px;}
.bn{border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:11px 15px;
margin:9px 0;font-size:13.5px;line-height:1.5;background:var(--panel);color:#3b4452;}
.bn.warn{border-left-color:#e8842a;}.bn.live{border-left-color:#2c6fb3;}.bn.keep{border-left-color:#2f9e44;}
nav.toc{border:1px solid var(--line);border-radius:12px;padding:14px 20px;margin:24px 0 10px;background:#fbfcfe;}
nav.toc h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
margin:0 0 8px;border:0;padding:0;}
nav.toc ol{margin:0;padding-left:20px;}nav.toc>ol>li{margin:6px 0;font-weight:600;}
nav.toc ol ol{padding-left:16px;}nav.toc ol ol li{margin:2px 0;font-weight:400;font-size:14px;}
nav.toc a{color:var(--accent);text-decoration:none;}nav.toc a:hover{text-decoration:underline;}
.chapter{border-top:2px solid var(--line);margin-top:40px;padding-top:6px;}
.ch-title{font-size:25px;margin:26px 0 14px;letter-spacing:-.01em;}
.ch-num{display:block;font-size:12px;font-weight:700;text-transform:uppercase;
letter-spacing:.08em;color:var(--accent);margin-bottom:3px;}
.chlede{color:var(--mut);font-size:15px;margin:2px 0 16px;}
.chapter h3{font-size:19px;margin:26px 0 8px;}
.chapter h4{font-size:16px;margin:20px 0 6px;color:var(--accent);}
.chapter h5{font-size:14.5px;margin:16px 0 5px;font-weight:600;}
.chapter ul,.chapter ol{margin:10px 0;}.chapter code{background:#f3f6fa;border:1px solid var(--line);
border-radius:4px;padding:1px 5px;font:13px/1.4 "SFMono-Regular",Consolas,Menlo,monospace;}
.book img,.chapter img{max-width:100%;height:auto}
.book figure,.chapter figure{max-width:100%;margin:1.2rem 0}
"""


def build_book(sl: dict) -> str:
    sections, toc_items = [], []
    num = 0
    for i, ch in enumerate(sl["chapters"], 1):
        if ch.get("unnumbered"):
            n = None
        else:
            num += 1
            n = num
        sec, anchors = build_chapter(sl["slug"], i, ch, n)
        sections.append(sec)
        toc_items.append(anchors)
    # TOC
    toc = ['<nav class="toc"><h2>Contents</h2><ol>']
    for anchors in toc_items:
        (cid, ctitle), subs = anchors[0], anchors[1:]
        toc.append(f'<li><a href="#{cid}">{ctitle}</a>')
        if subs:
            toc.append("<ol>")
            for a, t in subs:
                toc.append(f'<li><a href="#{a}">{t}</a></li>')
            toc.append("</ol>")
        toc.append("</li>")
    toc.append("</ol></nav>")
    mathjax = (
        '<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"],["$","$"]],'
        'displayMath:[["$$","$$"],["\\\\[","\\\\]"]]},svg:{fontCache:"global"}};</script>\n'
        '<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>TRINITY — {sl['title']}</title>\n{mathjax}\n"
        f'<style>{BOOK_CSS}</style></head><body><div class="book">\n'
        f"<h1>{sl['title']}</h1>\n<p class=\"lede\">{sl['subtitle']}</p>\n"
        f"{BANNERS}\n{''.join(toc)}\n{''.join(sections)}\n"
        "</div></body></html>"
    )


def main() -> None:
    for sl in STORYLINES:
        chapters = [c for c in sl["chapters"] if c.get("type") != "placeholder"]
        if not chapters:
            continue
        sl = {**sl, "chapters": chapters}
        out = HERE / f"storyline_{sl['slug']}.html"
        html = build_book(sl)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}  ({len(html)//1024} KB, {len(chapters)} ch)")


if __name__ == "__main__":
    main()
