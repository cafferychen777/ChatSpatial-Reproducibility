"""Generate figures/figure1.svg (Direction A, Ink & Cobalt), round 4 layout.

Page 180 x 117 mm. viewBox 1800 x 1170 units, so 10 units = 1 mm and 1 pt = 3.53 units.
Font sizes: 8 pt -> 28.2 (panel letters), 7 pt -> 24.7 (panel titles), 6.5 pt -> 22.9,
6 pt -> 21.2 (body), 5.5 pt -> 19.4 (smallest text).

Panel boxes (mm): a 0-180 x 0-48; b 0-64 x 53-117; c 68-120 x 53-117; d 124-180 x 53-117.
Export: inkscape figure1.svg --export-type=pdf --export-filename=figure1.pdf
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "figure1.svg"

# ---- palette (figure_style.py, Direction A) ----
COBALT, COBALT_MID, COBALT_LIGHT = "#1F5A96", "#6E97C4", "#C9DAEC"
ACCENT_FILL = "#EAF1F8"            # SCHEMATIC["accent_fill"]
RUST = "#B4513A"
INK, INK_SOFT = "#1A1A1A", "#5C5C5C"
HAIRLINE, NEUTRAL_LIGHT = "#C8C8C8", "#E4E4E4"
EDGE = "#A6A6A6"                    # neutral box outline (local tone between HAIRLINE and INK_SOFT)
ICON = "#4D4D4D"                    # SCHEMATIC["box_edge"]; monoline icon stroke

PT = (1800 + 2) / (180 / 25.4 * 72)   # units per pt (viewBox is 1802 units wide = 180 mm)
F_LET, F_TITLE, F_HEAD, F_BODY, F_SMALL = 8 * PT, 7 * PT, 6.5 * PT, 6 * PT, 5.5 * PT
SW_BOX, SW_ACC, SW_ICON, SW_ARROW, SW_THIN = 1.4, 1.8, 2.0, 1.8, 1.4
W, H = 1800, 1170
MONO = "Menlo, monospace"

parts: list[str] = []


def add(s: str) -> None:
    parts.append(s)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=F_BODY, weight="normal", fill=INK, anchor="start", italic=False, mono=False):
    fam = f' font-family="{MONO}"' if mono else ""
    st = ' font-style="italic"' if italic else ""
    wt = f' font-weight="{weight}"' if weight != "normal" else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:.2f}"{wt}{st}{fam} fill="{fill}" '
        f'text-anchor="{anchor}">{esc(s)}</text>')


def lines(x, y, ls, size=F_BODY, lh=None, **kw):
    lh = lh or size * 1.22
    for i, s in enumerate(ls):
        text(x, y + i * lh, s, size=size, **kw)


def rect(x, y, w, h, stroke=EDGE, fill="#FFFFFF", sw=SW_BOX, rx=8):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')


def path(d, stroke=ICON, sw=SW_ICON, fill="none", extra=""):
    add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')


def arrow(d, color=INK_SOFT, sw=SW_ARROW, both=False):
    mk = {COBALT: "arrC", RUST: "arrR"}.get(color, "arrG")
    st = f' marker-start="url(#{mk})"' if both else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="butt" '
        f'stroke-linejoin="round" marker-end="url(#{mk})"{st}/>')


def circle(cx, cy, r, stroke=ICON, fill="none", sw=SW_ICON):
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>')


def g_open(tx, ty, s=1.0, gid=None):
    i = f' id="{gid}"' if gid else ""
    add(f'<g{i} transform="translate({tx:.1f} {ty:.1f}) scale({s})">')


def g_close():
    add("</g>")


def badge(cx, cy, n, r=11.5):
    circle(cx, cy, r, stroke=COBALT, fill=COBALT, sw=0.8)
    text(cx, cy + 6.8, str(n), size=F_SMALL, weight="bold", fill="#FFFFFF", anchor="middle")


def panel_header(x, y, letter, title_lines, sub=None):
    """Letter + bold title. Every row-2 header occupies two text lines."""
    text(x, y, letter, size=F_LET, weight="bold")
    for i, s in enumerate(title_lines):
        text(x + 30, y + i * 30, s, size=F_TITLE, weight="bold")
    if sub:
        text(x + 30, y + len(title_lines) * 30, sub, size=F_BODY, fill=COBALT, weight="bold")


# ---------------------------------------------------------------- icons (40 x 40 box, monoline)
def ic_person(x, y, s=1.0, c=ICON):
    g_open(x, y, s)
    circle(20, 12, 8, stroke=c, sw=SW_ICON / s)
    path("M4 39c0-10 7-17 16-17s16 7 16 17", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_chip(x, y, s=1.0, c=ICON):
    g_open(x, y, s)
    rect(8, 8, 24, 24, stroke=c, sw=SW_ICON / s, rx=4)
    path("M15 2v6M20 2v6M25 2v6M15 32v6M20 32v6M25 32v6M2 15h6M2 20h6M2 25h6M32 15h6M32 20h6M32 25h6",
         stroke=c, sw=SW_ICON / s)
    path("M20 13.5c.7 4.3 1.9 5.6 6 6.5-4.1.9-5.3 2.2-6 6.5-.7-4.3-1.9-5.6-6-6.5 4.1-.9 5.3-2.2 6-6.5Z",
         stroke=c, sw=SW_ICON / s * 0.8)
    g_close()


def ic_shield(x, y, s=1.0, c=COBALT):
    g_open(x, y, s)
    path("M20 3 34 8v11c0 9-6 15-14 18C12 34 6 28 6 19V8Z", stroke=c, sw=SW_ICON / s)
    path("m13 20 5 5 9-10", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_clipboard(x, y, s=1.0, c=COBALT):
    g_open(x, y, s)
    path("M12 6H8v31h24V6h-4", stroke=c, sw=SW_ICON / s)
    path("M14 3h12v6H14Z", stroke=c, sw=SW_ICON / s)
    path("m13 18 3 3 5-6M24 18h5M13 29l3 3 5-6M24 29h5", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_cube(x, y, s=1.0, c=ICON):
    g_open(x, y, s)
    path("M20 3 35 11v18L20 37 5 29V11Z", stroke=c, sw=SW_ICON / s)
    path("M5 11l15 8 15-8M20 19v18", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_doc(x, y, s=1.0, c=COBALT):
    g_open(x, y, s)
    path("M8 3h17l8 8v26H8Z", stroke=c, sw=SW_ICON / s)
    path("M25 3v8h8M13 18h15M13 24h15M13 30h10", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_container(x, y, s=1.0, c=ICON):
    g_open(x, y, s)
    path("M3 11h34v22H3Z", stroke=c, sw=SW_ICON / s)
    path("M10 15v14M16 15v14M22 15v14M28 15v14M3 7h34", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_window(x, y, s=1.0, c=ICON):
    g_open(x, y, s)
    rect(3, 5, 34, 30, stroke=c, sw=SW_ICON / s, rx=4)
    path("M3 12h34M10 20h14M16 27h14", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_code(x, y, s=1.0, c=RUST):
    g_open(x, y, s)
    path("M13 11 4 20l9 9M27 11l9 9-9 9M23 7l-6 26", stroke=c, sw=SW_ICON / s)
    g_close()


def ic_xcircle(x, y, s=1.0, c=RUST):
    g_open(x, y, s)
    circle(20, 20, 15, stroke=c, sw=SW_ICON / s)
    path("M14 14l12 12M26 14 14 26", stroke=c, sw=SW_ICON / s)
    g_close()


BLOB = "M8 26C6 14 16 6 27 9c8-6 21-2 22 9 8 6 5 19-4 22-4 9-18 10-24 4C10 45 3 36 8 26Z"
CUTS = [
    "M27 9C23 22 30 32 24 44",
    "M8 26C20 22 34 30 53 24",
    "M18 12C28 20 26 34 40 40",
    "M10 36C22 30 30 18 46 14",
    "M38 8C30 20 38 30 30 44M8 30C18 28 24 32 34 30",
    "M14 14C22 28 36 24 50 34",
]


def ic_tissue_var(x, y, k, s=0.8):
    """Monoline tissue outline with a different (rust) partition per output."""
    g_open(x, y, s)
    path(BLOB, stroke=ICON, sw=1.6 / s)
    path(CUTS[k], stroke=RUST, sw=1.6 / s)
    g_close()


# =================================================================== document
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{(H + 2) * 180 / (W + 2):.3f}mm" '
    f'viewBox="-1 -1 {W + 2} {H + 2}" font-family="Arial, Helvetica, sans-serif">')
add('<title>ChatSpatial overview</title>')
add('<desc>From request to recorded result; typed schemas define an enumerable parameter space; '
    'one recorded workflow across Python and R; 65 methods in 13 categories.</desc>')
add('<defs>')
for mid_, colr in (("arrG", INK_SOFT), ("arrC", COBALT), ("arrR", RUST)):
    add(f'<marker id="{mid_}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" '
        f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0 0.8 10 5 0 9.2Z" fill="{colr}"/></marker>')
add('</defs>')
add(f'<rect x="-1" y="-1" width="{W + 2}" height="{H + 2}" fill="#FFFFFF"/>')


def mid(c):
    return (c[0] + c[1]) / 2


# =================================================================== panel a  (0-1800 x 0-520)
add('<g id="panel_a">')
text(0, 28, "a", size=F_LET, weight="bold")
text(30, 28, "From request to recorded result", size=F_TITLE, weight="bold")

# shared columns (1 mm grid)
C0 = (0, 210)        # user
C1 = (280, 650)      # LLM step
C2 = (710, 1020)     # free-form code / tool layer
C3 = (1080, 1300)    # tissue outlines / wrapper
C4 = (1360, 1580)    # unconstrained output / result + record
C5 = (1640, 1800)    # re-execution
A1 = 110             # upper (code-generation) lane axis
A2 = 300             # lower (ChatSpatial) lane axis

# ---- user request
ic_person(mid(C0) - 20, 170, 1.0)
BUB = (C0[0], 225, C0[1] - C0[0], 150)
rect(*BUB, stroke=EDGE, rx=10)
path(f"M{mid(C0) - 8} 225 {mid(C0)} 215 {mid(C0) + 8} 225", stroke=EDGE, sw=SW_BOX, fill="#FFFFFF")
add(f'<path d="M{mid(C0) - 6.5} 226h13" stroke="#FFFFFF" stroke-width="2.6"/>')
lines(20, 272, ["“Identify spatial", "domains in this", "section.”"], size=F_HEAD, lh=29)

# fork
FX = 245
arrow(f"M{C0[1]} {A2}H{C1[0] - 2}", color=COBALT)
arrow(f"M{FX} {A2}V{A1 + 12}Q{FX} {A1} {FX + 12} {A1}H{C1[0] - 2}", color=RUST, sw=SW_THIN)

# ---- upper lane: code generation (rust, thin, secondary)
ic_chip(C1[0], A1 - 20, 1.0, c=ICON)
text(C1[0] + 52, A1 + 7.5, "LLM writes code", size=F_BODY, fill=RUST, weight="bold")
arrow(f"M{C1[0] + 230} {A1}H{C2[0] - 2}", color=RUST, sw=SW_THIN)
rect(C2[0], A1 - 40, C2[1] - C2[0], 80, stroke=RUST, sw=SW_BOX)
ic_code(C2[0] + 14, A1 - 31, 0.62)
text(C2[0] + 48, A1 - 8, "Free-form code", size=F_BODY, weight="bold", fill=RUST)
text(C2[0] + 16, A1 + 26, "package APIs · imports · syntax", size=F_SMALL, fill=INK_SOFT)
arrow(f"M{C2[1]} {A1}H{C3[0] - 2}", color=RUST, sw=SW_THIN)
for k in range(6):
    ic_tissue_var(C3[0] + 22 + (k % 3) * 62, A1 - 42 + (k // 3) * 42, k, s=0.72)
arrow(f"M{C3[1]} {A1}H{C4[0] - 2}", color=RUST, sw=SW_THIN)
ic_xcircle(C4[0], A1 - 16, 0.8)
text(C4[0] + 42, A1 + 7.5, "Unconstrained output", size=F_BODY, fill=RUST, weight="bold")

# ---- lower lane: ChatSpatial (cobalt, primary)
BT, BH = 225, 150            # tall boxes (typed call, tool layer)
ST, SH = 240, 120            # step boxes
# typed tool call
rect(C1[0], BT, C1[1] - C1[0], BH, stroke=COBALT, fill=ACCENT_FILL, sw=SW_ACC)
ic_chip(C1[0] + 16, BT + 14, 0.8, c=COBALT)
text(C1[0] + 56, BT + 37, "LLM selects a typed tool call", size=F_BODY, weight="bold", fill=COBALT)
text(C1[0] + 18, BT + 78, "identify_spatial_domains(", size=F_SMALL, mono=True, fill=INK)
text(C1[0] + 18 + 2 * 11.7, BT + 102, 'method="spagcn",', size=F_SMALL, mono=True, fill=INK)
text(C1[0] + 18 + 2 * 11.7, BT + 126, "n_domains=7)", size=F_SMALL, mono=True, fill=INK)
arrow(f"M{C1[1]} {A2}H{C2[0] - 2}", color=COBALT)

# tool layer with two stacked checks
rect(C2[0], BT, C2[1] - C2[0], BH, stroke=COBALT, fill=ACCENT_FILL, sw=SW_ACC)
text(C2[0] + 16, BT + 30, "ChatSpatial tool layer", size=F_BODY, weight="bold", fill=COBALT)
for i, (icon, lab) in enumerate(((ic_shield, "Schema validation"), (ic_clipboard, "Precondition checks"))):
    yy = BT + 46 + i * 50
    rect(C2[0] + 14, yy, C2[1] - C2[0] - 28, 42, stroke=COBALT_MID, sw=1.2, rx=6)
    icon(C2[0] + 24, yy + 6, 0.75, COBALT)
    text(C2[0] + 62, yy + 28, lab, size=F_BODY, fill=INK)

# pass -> wrapper -> record -> re-execution


def step_box(col, icon_fn, labels, accent=False, bold_first=False, icol=ICON):
    x0, x1 = col
    rect(x0, ST, x1 - x0, SH, stroke=COBALT if accent else EDGE,
         fill=ACCENT_FILL if accent else "#FFFFFF", sw=SW_ACC if accent else SW_BOX)
    n = len(labels)
    block = 34 + 8 + n * 24
    top = ST + (SH - block) / 2
    icon_fn(mid(col) - 17, top, 0.85, icol)
    for i, s in enumerate(labels):
        text(mid(col), top + 42 + 17 + i * 24, s, size=F_BODY, anchor="middle",
             weight="bold" if (bold_first and i == 0) else "normal",
             fill=COBALT if (bold_first and i == 0) else INK)


arrow(f"M{C2[1]} {A2}H{C3[0] - 2}", color=COBALT)
text(mid((C2[1], C3[0])), A2 - 10, "pass", size=F_SMALL, fill=COBALT, anchor="middle")
step_box(C3, ic_cube, ["Versioned", "Python/R wrapper"])
arrow(f"M{C3[1]} {A2}H{C4[0] - 2}", color=COBALT)
step_box(C4, ic_doc, ["Result + run record", "tool · parameters ·", "data · versions"],
         accent=True, bold_first=True, icol=COBALT)
arrow(f"M{C4[1]} {A2}H{C5[0] - 2}", color=COBALT)
step_box(C5, ic_container, ["Re-execution", "without LLM"])

# badges 1 and 2 (below the elements where the properties act)
BLY = 406
badge(C1[0] + 12, BLY - 7, 1)
lines(C1[0] + 32, BLY, ["Typed interface sets", "which analysis runs"], size=F_BODY, lh=25, fill=INK)
badge(C2[0] + 72, BLY - 7, 2)
lines(C2[0] + 92, BLY, ["Tool layer enforces", "what an analysis may do"], size=F_BODY, lh=25, fill=INK)

# recorded calls re-execute without any LLM (no badge: not one of the three properties)
ic_container(C4[0] + 12 - 13, BLY - 7 - 13, 0.65, COBALT)
lines(C4[0] + 32, BLY, ["Public container image; reproduction", "checked with every release"],
      size=F_BODY, lh=25, fill=INK)

# fail branch: tool layer -> explicit refusal -> back to the user
RX_ = C2[0] + 36
RY = 466
arrow(f"M{RX_} {BT + BH}V{RY - 12}Q{RX_} {RY} {RX_ - 12} {RY}H{mid(C0) + 12}"
      f"Q{mid(C0)} {RY} {mid(C0)} {RY - 12}V{BUB[1] + BUB[3] + 2}", color=INK_SOFT, sw=SW_THIN)
text(RX_ - 9, BT + BH + 30, "fail", size=F_SMALL, fill=INK_SOFT, anchor="end")
LBX = 465
add(f'<rect x="{LBX - 152}" y="{RY - 16}" width="350" height="32" fill="#FFFFFF"/>')
ic_xcircle(LBX - 140, RY - 13, 0.65, c=INK_SOFT)
text(LBX - 108, RY + 7, "Explicit refusal with reason", size=F_BODY, weight="bold", fill=INK)
add('</g>')

# =================================================================== row 2 (y 580-1220)
RY0 = 530
TY = RY0 + 28          # first title baseline
CY0 = RY0 + 85         # content top, shared by b, c, d
CY1 = H                # content bottom

# ------------------------------------------------------------------- panel b (0-640)
add('<g id="panel_b">')
panel_header(0, TY, "b", ["Typed schemas define an", "enumerable parameter space"])
CARD = (0, CY0, 340, 380)
cx0, cy0, cw, ch = CARD
rect(cx0, cy0, cw, ch, stroke=COBALT, sw=SW_ACC, rx=8)
add(f'<path d="M{cx0 + 0.9} {cy0 + 44}V{cy0 + 8}Q{cx0 + 0.9} {cy0 + 0.9} {cx0 + 8} {cy0 + 0.9}'
    f'H{cx0 + cw - 8}Q{cx0 + cw - 0.9} {cy0 + 0.9} {cx0 + cw - 0.9} {cy0 + 8}V{cy0 + 44}Z" '
    f'fill="{ACCENT_FILL}" stroke="none"/>')
add(f'<path d="M{cx0} {cy0 + 44}H{cx0 + cw}" stroke="{COBALT_LIGHT}" stroke-width="1.2"/>')
text(cx0 + 16, cy0 + 29, "identify_spatial_domains", size=F_SMALL, mono=True, weight="bold", fill=COBALT)
fields = [
    ("method", ["one of 7: spagcn, stagate, graphst,", "banksy, leiden, louvain,"]),
    ("n_domains", ["integer, 1 to 50"]),
    ("resolution", ["number, > 0 and ≤ 5"]),
    ("refine_domains", ["true or false"]),
]
fy = cy0 + 80
for k, (name, cons) in enumerate(fields):
    text(cx0 + 16, fy, name, size=F_SMALL, mono=True, weight="bold", fill=INK)
    for j, s in enumerate(cons):
        text(cx0 + 16, fy + 24 + j * 23, s, size=F_SMALL, fill=INK_SOFT)
    if k == 0:   # seventh method: available, but not part of the robustness study
        add(f'<text x="{cx0 + 16:.1f}" y="{fy + 24 + len(cons) * 23:.1f}" font-size="{F_SMALL:.2f}" '
            f'fill="{EDGE}">aestetik <tspan font-style="italic">(not in robustness study)</tspan></text>')
        fy += 23
    fy += 24 + len(cons) * 23 + 21

# lattice: one column per method of the robustness study, height proportional to the number of
# configurations per section (experiments/robustness/PREREGISTRATION.md, 1.5.5 addendum: calls x 6
# post-processing variants; confirmed in experiments/robustness/results/configurations_dlpfc155.csv.gz). One dot = 30 configurations.
METHOD_COLORS = {"SpaGCN": "#2E4A6B", "STAGATE": "#5E9AAE", "GraphST": "#C2913A",
                 "BANKSY": RUST, "Leiden": "#C4C4C4", "Louvain": "#5A5A5A"}   # figure_style.py
N_CONF = {"SpaGCN": 210, "STAGATE": 210, "GraphST": 504, "BANKSY": 900, "Leiden": 450, "Louvain": 450}
assert sum(N_CONF.values()) == 2724
LX0, LX1 = 400, 640
CPITCH, DP, DR, PER_DOT = 41, 12, 4.8, 30
BL = 880                                     # centre of the bottom dot row
c_first = mid((LX0, LX1)) - (len(N_CONF) - 1) * CPITCH / 2
text(mid((LX0, LX1)), cy0 + 22, "Every valid", size=F_BODY, weight="bold", anchor="middle")
text(mid((LX0, LX1)), cy0 + 47, "configuration", size=F_BODY, weight="bold", anchor="middle")
for m, (meth, n) in enumerate(N_CONF.items()):
    cxm = c_first + m * CPITCH
    nd = round(n / PER_DOT)
    for d in range(nd):
        r_, c_ = divmod(d, 3)
        add(f'<circle cx="{cxm + (c_ - 1) * DP:.1f}" cy="{BL - r_ * DP:.1f}" r="{DR}" '
            f'fill="{METHOD_COLORS[meth]}"/>')
    top = BL - ((nd - 1) // 3) * DP - DR
    add(f'<text transform="translate({cxm:.1f} {top - 8:.1f}) rotate(-90)" x="0" y="7" '
        f'font-size="{F_SMALL:.2f}" fill="{INK}">{meth}</text>')
    text(cxm, BL + DR + 22, f"{n}", size=F_SMALL, anchor="middle", fill=INK_SOFT)
arrow(f"M{cx0 + cw + 6} {BL - DP}H{c_first - 22}", color=COBALT)
lines(mid((LX0, LX1)), BL + 62, ["2,724 configurations per", "section for the six methods",
                                  "of the robustness study", "(Fig. 4)"],
      size=F_SMALL, lh=22, anchor="middle", fill=INK)

# badge 3
B3Y = cy0 + ch + 46
badge(12, B3Y - 7, 3)
text(32, B3Y, "Enumerable parameter space makes robustness systematic", size=F_BODY)

# 449-parameter bar
PB_Y = CY1 - 52
text(0, PB_Y - 14, "449 parameters across 20 tools", size=F_BODY, weight="bold")
frac = 375 / 449
add(f'<rect x="0" y="{PB_Y}" width="{640 * frac:.1f}" height="20" fill="{COBALT}"/>')
add(f'<rect x="{640 * frac + 2:.1f}" y="{PB_Y}" width="{640 * (1 - frac) - 2:.1f}" height="20" fill="{NEUTRAL_LIGHT}"/>')
text(0, PB_Y + 46, "375 constrained (83.5%)", size=F_SMALL, fill=COBALT, weight="bold")
text(640, PB_Y + 46, "74 free text", size=F_SMALL, fill=INK_SOFT, anchor="end")
add('</g>')

# ------------------------------------------------------------------- panel c (680-1200)
add('<g id="panel_c">')
PC = (680, 1200)
panel_header(PC[0], TY, "c", ["One recorded workflow", "across Python and R"])
XC = mid(PC)
# client + LLM
CL = (XC - 150, CY0, 300, 70)
rect(*CL, rx=8)
ic_chip(CL[0] + 16, CL[1] + 15, 1.0)
text(CL[0] + 66, CL[1] + 31, "MCP client + LLM", size=F_BODY)
text(CL[0] + 66, CL[1] + 55, "Model-agnostic", size=F_SMALL, italic=True, fill=INK_SOFT)
arrow(f"M{XC} {CL[1] + CL[3]}V{CL[1] + CL[3] + 38}", color=COBALT)
# server
SV = (PC[0], CL[1] + CL[3] + 40, PC[1] - PC[0], 62)
rect(*SV, stroke=COBALT, fill=ACCENT_FILL, sw=SW_ACC)
text(XC, SV[1] + 40, "ChatSpatial server", size=F_BODY, weight="bold", fill=COBALT, anchor="middle")
# environments
EW, EG = 230, 60
PX = (PC[0], PC[0] + EW)
RX = (PC[1] - EW, PC[1])
EY, EH = SV[1] + SV[3] + 44, 160
arrow(f"M{mid(PX)} {SV[1] + SV[3]}V{EY - 2}", color=COBALT)
arrow(f"M{mid(RX)} {SV[1] + SV[3]}V{EY - 2}", color=COBALT)


def env_box(col, title, items):
    x0, x1 = col
    rect(x0, EY, x1 - x0, EH, rx=8)
    text(x0 + 14, EY + 32, title, size=F_BODY, weight="bold")
    add(f'<path d="M{x0 + 14} {EY + 46}H{x1 - 14}" stroke="{NEUTRAL_LIGHT}" stroke-width="1.4"/>')
    for i, s in enumerate(items):
        text(x0 + 14, EY + 80 + i * 27, s, size=F_BODY)


env_box(PX, "Python", ["AnnData", "Scanpy", "Squidpy"])
env_box(RX, "R", ["Seurat", "CellChat", "SingleCellExperiment"])
# rpy2: two-way link in the gap between the Python and R boxes
BRY = EY + EH / 2 + 12
arrow(f"M{PX[1] + 3} {BRY}H{RX[0] - 3}", color=COBALT, both=True)
add(f'<text x="{XC:.1f}" y="{BRY - 12:.1f}" font-size="{F_SMALL:.2f}" font-family="{MONO}" '
    f'fill="{COBALT}" text-anchor="middle">rpy2</text>')
# run record strip; each environment records its own calls
RR = (PC[0], CY1 - 56, PC[1] - PC[0], 56)
arrow(f"M{mid(PX)} {EY + EH}V{RR[1] - 2}", color=COBALT)
arrow(f"M{mid(RX)} {EY + EH}V{RR[1] - 2}", color=COBALT)
rect(*RR, stroke=COBALT, fill=ACCENT_FILL, sw=SW_ACC)
ic_doc(RR[0] + 14, RR[1] + 10, 0.9, COBALT)
text(RR[0] + 58, RR[1] + 35, "Run record", size=F_BODY, weight="bold", fill=COBALT)
text(RR[0] + 178, RR[1] + 35, "every call, both languages", size=F_SMALL, fill=INK_SOFT)
add('</g>')

# ------------------------------------------------------------------- panel d (1240-1800)
add('<g id="panel_d">')
PD = (1240, 1800)
panel_header(PD[0], TY, "d", ["65 methods in 13 categories"], sub="20 typed tools")
cats = [
    ("Spatial statistics", 13), ("Deconvolution", 8), ("Spatial domain identification", 7),
    ("Cell type annotation", 6), ("Differential expression", 6), ("Enrichment analysis", 5),
    ("Multi-sample integration", 4), ("Cell–cell communication", 4),
    ("Spatially variable genes", 3), ("Trajectory inference", 3), ("Spatial registration", 2),
    ("CNV analysis", 2), ("RNA velocity", 2),
]
assert sum(n for _, n in cats) == 65 and len(cats) == 13
R0, R1 = CY0 + 12, CY1 - 12           # first / last row centre
PITCH = (R1 - R0) / (len(cats) - 1)
BAR_X, BAR_MAX, BAR_H = PD[0] + 290, 230, 20
add(f'<path d="M{BAR_X} {R0 - PITCH / 2:.1f}V{R1 + BAR_H / 2 + 2:.1f}" stroke="{INK_SOFT}" stroke-width="1.2"/>')
for k, (lab, n) in enumerate(cats):
    yc = R0 + k * PITCH
    text(BAR_X - 12, yc + 7.3, lab, size=F_BODY, anchor="end")
    bw = BAR_MAX * n / 13
    add(f'<rect x="{BAR_X}" y="{yc - BAR_H / 2:.1f}" width="{bw:.1f}" height="{BAR_H}" fill="{COBALT}"/>')
    text(BAR_X + bw + 8, yc + 7.3, str(n), size=F_BODY, fill=INK)
add('</g>')

add("</svg>")
OUT.write_text("\n".join(parts) + "\n")
print("wrote", OUT)
