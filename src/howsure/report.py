"""A single self-contained HTML report: one card per question, two charts each.

1. "Does the model mean what it says?" Stated confidence vs. how often it was right,
   before and after calibration (a reliability diagram).
2. "How much can you automate?" Errors among auto-handled cases vs. how many cases
   that covers, with the target line and both cutoffs marked.
"""

from __future__ import annotations

import html
import json
import math

from .policy import Policy, describe

W, H = 440, 290
ML, MR, MT, MB = 50, 18, 14, 44

_CSS = """
:root{color-scheme:light;
--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--good:#006300;--warn-bg:#fff4d6;--warn-ink:#6b4a00}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--warn-bg:#3a2e10;--warn-ink:#f3d38a}}
:root[data-theme="dark"]{color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--warn-bg:#3a2e10;--warn-ink:#f3d38a}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:0}
.sub{color:var(--ink2);margin:0 0 24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;margin:0 0 20px}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.tag{color:var(--ink2);font-size:13px}
ul.facts{margin:0 0 14px;padding-left:18px}
ul.facts li{margin:2px 0}
.charts{display:grid;grid-template-columns:1fr;gap:16px}
@media (min-width:760px){.charts{grid-template-columns:1fr 1fr}}
figure{margin:0;min-width:0}
.scroll{overflow-x:auto}
figcaption{font-size:14px;font-weight:600;margin:0 0 2px}
.hint{font-size:13px;color:var(--ink2);margin:0 0 6px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:0 0 4px}
.key{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px;border-radius:1px}
svg{width:100%;height:auto;display:block;overflow:visible}
svg text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--muted)}
svg .lbl{fill:var(--ink2)}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--ink2);stroke-width:1;opacity:.55}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}
.hit{fill:transparent;cursor:default}
.xh{stroke:var(--ink2);stroke-width:1;opacity:0}
.xh.on{opacity:.5}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
border:1px solid var(--border);border-radius:8px;padding:6px 10px;font-size:13px;
box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9;max-width:260px}
#tip b{font-weight:600}
#tip .row{display:flex;gap:8px;align-items:center}
#tip .t2{color:var(--ink2)}
details{margin-top:12px}
summary{cursor:pointer;color:var(--ink2);font-size:14px}
table{border-collapse:collapse;margin-top:8px;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:4px 12px 4px 0;text-align:left;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:500}
.overview td,.overview th{padding-right:20px}
.note{background:var(--warn-bg);color:var(--warn-ink);border-radius:8px;padding:8px 12px;margin:8px 0;font-size:14px}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--page);padding:1px 5px;border-radius:4px}
pre{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--page);padding:12px;border-radius:8px;
overflow-x:auto}
"""

_JS = """
(function(){
var tip=document.getElementById('tip');
function show(e,rows){tip.textContent='';rows.forEach(function(r,i){var d=document.createElement('div');
 if(i===0){var b=document.createElement('b');b.textContent=r;d.appendChild(b);}else{d.className='t2';d.textContent=r;}
 tip.appendChild(d);});tip.style.display='block';move(e);}
function move(e){var x=e.clientX+14,y=e.clientY+14,w=tip.offsetWidth,h=tip.offsetHeight;
 if(x+w>window.innerWidth-8)x=e.clientX-w-14; if(y+h>window.innerHeight-8)y=e.clientY-h-14;
 tip.style.left=x+'px';tip.style.top=y+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('.hit[data-tip]').forEach(function(el){
 var rows=JSON.parse(el.getAttribute('data-tip'));
 el.addEventListener('pointermove',function(e){show(e,rows);});
 el.addEventListener('pointerleave',hide);
 el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show({clientX:r.right,clientY:r.top},rows);});
 el.addEventListener('blur',hide);});
document.querySelectorAll('svg[data-curve]').forEach(function(svg){
 var pts=JSON.parse(svg.getAttribute('data-curve')),xh=svg.querySelector('.xh'),dot=svg.querySelector('.xdot');
 var area=svg.querySelector('.area');
 area.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal;
  var px=(e.clientX-r.left)*vb.width/r.width,best=pts[0];
  for(var i=0;i<pts.length;i++){if(Math.abs(pts[i][0]-px)<Math.abs(best[0]-px))best=pts[i];}
  xh.setAttribute('x1',best[0]);xh.setAttribute('x2',best[0]);xh.classList.add('on');
  dot.setAttribute('cx',best[0]);dot.setAttribute('cy',best[1]);dot.style.opacity=1;
  show(e,[best[3]+' errors','handles '+best[2]+' of cases','auto when p \\u2265 '+best[4]]);});
 area.addEventListener('pointerleave',function(){xh.classList.remove('on');dot.style.opacity=0;hide();});});
})();
"""


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _pct(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.{digits}f}%"


def _tip(rows: list[str]) -> str:
    return _esc(json.dumps(rows))


def _y_max(v: float) -> float:
    for c in (0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.32, 0.4, 0.6, 0.8, 1.0):
        if v <= c:
            return c
    return 1.0


def _reliability_svg(before: list[dict], after: list[dict]) -> str:
    pts = before + after
    lo = min([0.5] + [min(p["confidence"], p["accuracy"]) for p in pts])
    lo = max(0.0, math.floor(lo * 10) / 10)
    step = 0.1 if 1 - lo <= 0.5 else 0.2
    pw, ph = W - ML - MR, H - MT - MB

    def sx(v: float) -> float:
        return ML + (v - lo) / (1 - lo) * pw

    def sy(v: float) -> float:
        return MT + (1 - (v - lo) / (1 - lo)) * ph

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Stated confidence versus actual accuracy">']
    t = lo
    while t <= 1.0001:
        x, y = sx(t), sy(t)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>')
        out.append(f'<text x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle">{t * 100:.0f}%</text>')
        t = round(t + step, 6)
    out.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    out.append(f'<line class="ref" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}"/>')
    out.append(f'<text x="{ML + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">model says</text>')
    out.append(
        f'<text x="12" y="{MT + ph / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 12 {MT + ph / 2:.1f})">actually right</text>'
    )
    for series, color, name in ((before, "var(--s2)", "Before"), (after, "var(--s1)", "After")):
        if not series:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(p['confidence']):.1f},{sy(p['accuracy']):.1f}" for i, p in enumerate(series)
        )
        out.append(f'<path class="line" d="{d}" stroke="{color}"/>')
        for p in series:
            cx, cy = sx(p["confidence"]), sy(p["accuracy"])
            out.append(f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}"/>')
            tip = _tip(
                [
                    f"right {_pct(p['accuracy'])} of the time",
                    f"{name.lower()}: model said {_pct(p['confidence'])}",
                    f"{p['count']} examples",
                ]
            )
            out.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12" tabindex="0" data-tip="{tip}"/>')
    out.append("</svg>")
    return "".join(out)


def _risk_svg(chart: dict, max_error: float, sure: float) -> str:
    curve = [c for c in chart["curve"] if c[0] >= 0.02] or chart["curve"]
    ymax = _y_max(max(max(e for _, e, _ in curve), max_error * 1.6))
    pw, ph = W - ML - MR, H - MT - MB

    def sx(v: float) -> float:
        return ML + v * pw

    def sy(v: float) -> float:
        return MT + (1 - v / ymax) * ph

    pts = [[round(sx(c), 1), round(sy(e), 1), _pct(c), _pct(e, 1), f"{t:.3f}"] for c, e, t in curve]
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Errors versus share of cases handled automatically" '
        f'data-curve="{_esc(json.dumps(pts))}">'
    ]
    for i in range(5):
        v = ymax * i / 4
        y = sy(v)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{v * 100:g}%</text>')
    for i in range(5):
        x = sx(i / 4)
        out.append(f'<text x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle">{i * 25}%</text>')
    out.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    ty = sy(max_error)
    mark_xs = [sx(p[0]) for p in (chart.get("cutoff_point"), chart.get("safe_point")) if p]
    left_busy = any(x < ML + pw * 0.45 for x in mark_xs)
    tx, anchor = (W - MR - 6, "end") if left_busy else (ML + 6, "start")
    out.append(
        f'<line class="ref" x1="{ML}" x2="{W - MR}" y1="{ty:.1f}" y2="{ty:.1f}"/>'
        f'<text class="lbl" x="{tx}" y="{ty + 16:.1f}" text-anchor="{anchor}">'
        f"target {_pct(max_error, 1 if max_error < 0.1 else 0)}</text>"
    )
    out.append(f'<text x="{ML + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">share handled automatically</text>')
    out.append(
        f'<text x="12" y="{MT + ph / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 12 {MT + ph / 2:.1f})">errors among them</text>'
    )
    d = " ".join(f"{'M' if i == 0 else 'L'}{p[0]},{p[1]}" for i, p in enumerate(pts))
    out.append(f'<path class="line" d="{d}" stroke="var(--s1)"/>')

    marks = []
    if chart.get("cutoff_point"):
        c, e = chart["cutoff_point"]
        marks.append((sx(c), sy(e), f"auto-handle: {_pct(c)}"))
    if chart.get("safe_point"):
        c, e = chart["safe_point"]
        marks.append((sx(c), sy(e), f"{_pct(sure)} sure: {_pct(c)}"))
    for i, (x, y, label) in enumerate(marks):
        out.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="var(--s1)"/>')
        close = len(marks) == 2 and abs(marks[0][0] - marks[1][0]) < 90
        below = close and i == 1
        anchor = "end" if x > ML + pw * 0.6 else "start"
        dx = -8 if anchor == "end" else 8
        ly = y + 18 if below else y - 10
        out.append(f'<text class="lbl" x="{x + dx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{_esc(label)}</text>')
    out.append(
        f'<line class="xh" x1="0" x2="0" y1="{MT}" y2="{MT + ph}"/>'
        f'<circle class="xdot dot" r="4.5" fill="var(--s1)" style="opacity:0"/>'
        f'<rect class="area" x="{ML}" y="{MT}" width="{pw}" height="{ph}" fill="transparent"/>'
    )
    out.append("</svg>")
    return "".join(out)


def _facts(q: dict, max_error: float, sure: float) -> list[str]:
    s = q["stats"]
    facts = [
        f"Accuracy {_pct(s['accuracy_after'])}. Calibration error {_pct(s['ece_before'], 1)} → "
        f"{_pct(s['ece_after'], 1)} after the fix ({describe(q['calibration'])})."
    ]
    if q["cutoff"] is None:
        facts.append(f"No cutoff keeps errors within {_pct(max_error, 1)}: send this question to review.")
    else:
        facts.append(
            f"<b>Auto-handle when p ≥ {q['cutoff']:.3f}</b>: {_pct(s['coverage'])} of cases, "
            f"with {_pct(s['error'], 1)} errors on held-out examples."
        )
    if q["safe_cutoff"] is not None:
        facts.append(
            f"{_pct(sure)} sure version: p ≥ {q['safe_cutoff']:.3f} handles {_pct(s['safe_coverage'])} of cases, "
            f"and the true error rate is at most {_pct(max_error, 1)} with {_pct(sure)} confidence."
        )
    else:
        facts.append(f"Not enough examples yet to promise {_pct(max_error, 1)} with {_pct(sure)} confidence.")
    more = s.get("safe_coverage_with_more")
    if more and q["cutoff"] is not None:
        facts.append(
            f"More labels would help: with about {2 * s['n']:,} you could expect a {_pct(sure)}-sure cutoff "
            f"covering ~{_pct(more['2'])}; with {4 * s['n']:,}, ~{_pct(more['4'])}."
        )
    if s.get("score_rounding_mismatch"):
        facts.append(
            f"Rounding the model's <code>score</code> field disagrees with its most likely level in "
            f"{_pct(s['score_rounding_mismatch'])} of cases. Act on <code>howsure.answer</code> instead."
        )
    other = s["uncommon"]
    facts.append(
        f"Always answering <code>{_esc(s['common_answer'])}</code> would be right {_pct(s['common_accuracy'])} "
        f"of the time. The other {other['n']:,} held-out cases: {other['right']:,} right, {other['wrong']:,} wrong, "
        f"{other['review']:,} to review."
    )
    if other["lost_to_calibration"] or other["gained_from_calibration"]:
        facts.append(
            f"Calibration changed {other['lost_to_calibration']:,} of those from right to wrong, "
            f"and {other['gained_from_calibration']:,} from wrong to right."
        )
    return facts


def _groups_table(s: dict) -> str:
    groups = s.get("groups")
    if not groups:
        return ""
    body = "".join(
        f"<tr><td>{_esc(g['group'])}</td><td>{g['n']:,}</td><td>{_pct(g['coverage'])} at {_pct(g['error'], 1)}</td>"
        f"<td>{_pct(g.get('unseen_coverage'))} at {_pct(g.get('unseen_error'), 1)}</td></tr>"
        for g in groups
    )
    return (
        "<details><summary>By document type</summary><table><tr><th></th><th>Examples</th><th>This policy</th>"
        f"<th>Fitted without this type</th></tr>{body}</table>"
        '<p class="hint">Share auto-handled, and errors among those. The last column is the one to trust for '
        "documents the policy has never seen.</p></details>"
    )


def _table(s: dict) -> str:
    rows = [
        ("Accuracy", s["accuracy_before"], s["accuracy_after"]),
        ("Calibration error (ECE)", s["ece_before"], s["ece_after"]),
        ("Brier score", s["brier_before"], s["brier_after"]),
        ("Wrong with ≥ 90% confidence", s["confident_errors_before"], s["confident_errors_after"]),
    ]
    body = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_pct(a, 1) if 'Brier' not in name else f'{a:.3f}'}</td>"
        f"<td>{_pct(b, 1) if 'Brier' not in name else f'{b:.3f}'}</td></tr>"
        for name, a, b in rows
    )
    split = (
        f"Fitted on {s['n_fit']} examples, checked on {s['n_check']} others."
        if s["held_out"]
        else f"Only {s['n']} examples, so it was fitted and checked on the same ones: numbers are optimistic."
    )
    return (
        "<details><summary>Numbers</summary>"
        "<table><tr><th></th><th>Before</th><th>After</th></tr>"
        f'{body}</table><p class="hint">{_esc(split)}</p></details>'
    )


def render(policy: Policy, charts: dict, notes: list[str], policy_path: str = "policy.json") -> str:
    d = policy.data
    max_error, sure = d["target"]["max_error"], d["target"]["sure"]
    title = f"howsure · {d['model']}"
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>Can you trust {_esc(d['model'])}?</h1>",
        f'<p class="sub">Target: at most {_pct(max_error, 1)} errors on auto-handled cases · '
        f"{d['data']['examples']:,} labeled examples · {_esc(d['created'])}</p>",
    ]

    overview = "".join(
        f"<tr><td><b>{_esc(qid)}</b></td><td>{_esc(q['type'])}</td>"
        f"<td>{_pct(q['stats']['accuracy_after'])}</td>"
        f"<td>{_pct(q['stats']['ece_before'], 1)} → {_pct(q['stats']['ece_after'], 1)}</td>"
        f"<td>{_pct(q['stats']['coverage'])}</td><td>{_pct(q['stats']['safe_coverage'])}</td></tr>"
        for qid, q in d["questions"].items()
    )
    parts.append(
        '<section class="card"><div class="scroll"><table class="overview"><tr><th>Question</th><th>Type</th>'
        f"<th>Accuracy</th><th>Calibration error</th><th>Auto-handled</th><th>Auto-handled, {_pct(sure)} sure</th></tr>"
        f"{overview}</table></div></section>"
    )
    for n in notes:
        parts.append(f'<p class="note">{_esc(n)}</p>')

    for qid, q in d["questions"].items():
        ch = charts.get(qid)
        facts = "".join(f"<li>{f}</li>" for f in _facts(q, max_error, sure))
        parts.append(
            f'<section class="card"><div class="head"><h2>{_esc(qid)}</h2>'
            f'<span class="tag">{_esc(q["type"])} · {q["stats"]["n"]:,} labeled</span></div>'
            + "".join(f'<p class="note">{_esc(w)}</p>' for w in q["stats"]["warnings"])
            + f'<ul class="facts">{facts}</ul>'
        )
        if ch:
            parts.append(
                '<div class="charts"><figure><figcaption>Does it mean what it says?</figcaption>'
                '<p class="hint">When the model said this…, it was right this often. On the gray line = honest.</p>'
                '<div class="legend"><span><i class="key" style="background:var(--s2)"></i>Before</span>'
                '<span><i class="key" style="background:var(--s1)"></i>After calibration</span></div>'
                f"{_reliability_svg(ch['reliability_before'], ch['reliability_after'])}</figure>"
                "<figure><figcaption>How much can you automate?</figcaption>"
                '<p class="hint">Lower the cutoff to automate more, and errors rise. Hover to explore.</p>'
                f"{_risk_svg(ch, max_error, sure)}</figure></div>"
            )
        parts.append(_table(q["stats"]) + _groups_table(q["stats"]) + "</section>")

    first = _esc(next(iter(d["questions"])))
    pfile = _esc(policy_path)
    parts.append(
        '<section class="card"><h2>Use it</h2>'
        "<p>Add this policy to a <code>howsure gateway</code> config and point your app at the gateway. Each "
        "answer gets a <code>howsure</code> block with the calibrated probability and "
        "<code>action: auto | review</code>.</p>"
        f'<pre>{{"models": {{"model": {{"url": "http://localhost:8000", "policy": "{pfile}"}}}}, '
        '"default": "model"}\n\nhowsure gateway gateway.json</pre>'
        "<p>Or in Python:</p>"
        f'<pre>import howsure\npolicy = howsure.load("{pfile}")\n'
        f'decision = policy.decide(response)["{first}"]\nif decision.auto: ...</pre></section>'
    )
    parts.append(f'<div id="tip" role="tooltip"></div><script>{_JS}</script></main></body></html>')
    return "".join(parts)


def write(path: str, policy: Policy, charts: dict, notes: list[str], policy_path: str = "policy.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(policy, charts, notes, policy_path))
