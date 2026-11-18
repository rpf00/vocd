"""
SI tables (LaTeX booktabs style, rendered to PNG + PDF) — environmental
monitoring:

  Supplementary Table 3  CPs placed by each detector at the operating points of
                         Fig. 5 (from env_v3_counts_table.csv)
  Supplementary Table 4  direction and persistence of every CP on TP / TN that
                         matches a storm (from env_v3_direction.csv)

Both CSVs are written by examples/env_sensitivity_v3.py; nothing is recomputed.
Captions go in the SI document, so the images hold the table body only.

Style: booktabs rules (heavy top / bottom rule, light mid rule, a cmidrule
under spanning headers, no vertical lines), Computer Modern text (the font
matplotlib ships as cmr10) or --font sans for Arial, as in the figures.

    python examples/vocd_tableS_env.py
    python examples/vocd_tableS_env.py --font sans --font-pt 8
"""
from __future__ import annotations

import argparse
import csv
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "results", "env")
OUTDIR = os.path.join(_REPO, "results", "env", "figures")

LABEL = {"vocd": "VOCD", "bocd": "BOCD", "dm-bocd": "Dm-BOCD", "focus": "FOCuS",
         "np-focus": "NP-FOCuS", "nunc": "NUNC", "chad": "CHAD", "champ": "CHAMP"}
ORDER = list(LABEL)
MINUS = "\u2212"


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def set_font(kind):
    import matplotlib as mpl
    if kind == "serif":
        mpl.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
                             "mathtext.fontset": "cm", "axes.unicode_minus": False})
    else:
        mpl.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Arimo", "Liberation Sans",
                                                 "Helvetica", "DejaVu Sans"],
                             "mathtext.fontset": "dejavusans"})
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def fix(s, kind):
    """cmr10 has no unicode minus: use the math minus there."""
    s = str(s)
    if kind == "serif" and MINUS in s:
        s = s.replace(MINUS, "$-$")
    return s


# ── generic booktabs renderer ─────────────────────────────────────────────────
def render(header, rows, align, path, kind, spans=(), font_pt=9.0, pad_pt=9.0,
           row_gap=1.55, margin_pt=6.0):
    """header: list of column labels (may contain '\\n'); rows: list of lists of
    str; align: 'l' / 'c' / 'r' per column; spans: (label, first_col, last_col)
    drawn above the header with a cmidrule."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    header = [fix(h, kind) for h in header]
    rows = [[fix(c, kind) for c in r] for r in rows]
    spans = [(fix(l, kind), a, b) for l, a, b in spans]

    probe = plt.figure(figsize=(4, 4), dpi=300)
    rend = probe.canvas.get_renderer()

    def width(s):
        t = probe.text(0, 0, s, fontsize=font_pt)
        w = t.get_window_extent(rend).width * 72.0 / probe.dpi
        t.remove()
        return w

    ncol = len(header)
    colw = [0.0] * ncol
    for j in range(ncol):
        cells = header[j].split("\n") + [r[j] for r in rows]
        colw[j] = max(width(c) for c in cells) + 2 * pad_pt
    for lab, a, b in spans:                         # a span may widen its columns
        need = width(lab) + 2 * pad_pt - sum(colw[a:b + 1])
        if need > 0:
            for j in range(a, b + 1):
                colw[j] += need / (b - a + 1)
    plt.close(probe)

    line = font_pt * row_gap
    n_head = max(h.count("\n") + 1 for h in header)
    span_h = line if spans else 0.0
    W = sum(colw) + 2 * margin_pt
    H = 2 * margin_pt + span_h + n_head * line + 0.5 * line + len(rows) * line + 0.4 * line
    fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=300)
    X = lambda x: x / W
    Y = lambda y: 1.0 - y / H                       # y measured from the top, in pt
    xs = [margin_pt + sum(colw[:j]) for j in range(ncol)]

    def put(j, y, s, al):
        if al == "l":
            x, ha = xs[j] + pad_pt, "left"
        elif al == "r":
            x, ha = xs[j] + colw[j] - pad_pt, "right"
        else:
            x, ha = xs[j] + colw[j] / 2, "center"
        fig.text(X(x), Y(y), s, fontsize=font_pt, ha=ha, va="center")

    def rule(y, x0, x1, lw):
        fig.add_artist(Line2D([X(x0), X(x1)], [Y(y), Y(y)], lw=lw, color="black",
                              solid_capstyle="butt"))

    heavy, light = 0.9, 0.5
    y = margin_pt
    rule(y, margin_pt, W - margin_pt, heavy)                        # \toprule
    y += 0.35 * line
    if spans:
        for lab, a, b in spans:
            fig.text(X(xs[a] + sum(colw[a:b + 1]) / 2), Y(y + 0.5 * line), lab,
                     fontsize=font_pt, ha="center", va="center")
            rule(y + 0.95 * line, xs[a] + pad_pt / 2, xs[b] + colw[b] - pad_pt / 2, light)  # \cmidrule
        y += span_h
    for j, h in enumerate(header):                                  # bottom-aligned
        parts = h.split("\n")
        for k, p in enumerate(parts):
            put(j, y + (n_head - len(parts) + k + 0.5) * line, p,
                "c" if align[j] != "l" else "l")
    y += n_head * line + 0.15 * line
    rule(y, margin_pt, W - margin_pt, light)                        # \midrule
    y += 0.2 * line
    for r in rows:
        for j, c in enumerate(r):
            put(j, y + 0.5 * line, c, align[j])
        y += line
    y += 0.2 * line
    rule(y, margin_pt, W - margin_pt, heavy)                        # \bottomrule

    fig.savefig(path + ".png", dpi=300)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    print(f"wrote {path}.png and {path}.pdf  ({W / 72 * 25.4:.0f} x {H / 72 * 25.4:.0f} mm)")


# ── the two tables ────────────────────────────────────────────────────────────
def table1(path_csv, out, kind, font_pt):
    rows = sorted(load(path_csv), key=lambda r: ORDER.index(r["method"]))
    body = [[LABEL[r["method"]], r["cps_positives"], r["cps_controls"],
             r["cps_controls_storm_adjacent"], r["cps_controls_away_from_storms"],
             f"{r['controls_with_storm_adjacent_cp']}/{r['n_controls']}"] for r in rows]
    header = ["Method", "CPs on\npositives", "Total", "Storm-\nadjacent", "Away from\nstorms",
              "Controls\nflagged"]
    render(header, body, ["l", "c", "c", "c", "c", "c"], out, kind,
           spans=[("CPs on controls", 2, 4)], font_pt=font_pt)


def table2(path_csv, out, kind, font_pt):
    rows = sorted(load(path_csv), key=lambda r: (ORDER.index(r["method"]), r["cp_date"]))
    sg = lambda v, d: (f"{float(v):+.{d}f}" if d else f"{float(v):+.0f}").replace("-", MINUS)
    body = []
    for r in rows:
        lag = int(float(r["lag_days"]))
        body.append([LABEL[r["method"]], r["constituent"], r["cp_date"], r["event"],
                     (f"{lag:+d}" if lag else "0").replace("-", MINUS),
                     f"{float(r['median_before']):.3f}", f"{float(r['median_after']):.3f}",
                     sg(r["change_pct"], 0),
                     "up" if str(r["positive_shift"]).strip().lower() == "true" else "down"])
    unit = "(mg L$^{-1}$)"
    header = ["Method", "Constituent", "CP date", "Storm", "Lag\n(days)",
              f"Before\n{unit}", f"After\n{unit}", "Change\n(%)", "Shift"]
    render(header, body, ["l", "l", "l", "l", "r", "r", "r", "r", "c"], out, kind,
           spans=[("30-day median", 5, 6)], font_pt=font_pt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=os.path.join(RES, "env_v3_counts_table.csv"))
    ap.add_argument("--direction", default=os.path.join(RES, "env_v3_direction.csv"))
    ap.add_argument("--font", choices=["serif", "sans"], default="serif",
                    help="serif = Computer Modern (LaTeX look); sans = Arial, as the figures")
    ap.add_argument("--font-pt", type=float, default=9.0)
    ap.add_argument("--out-dir", default=OUTDIR)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    set_font(args.font)
    os.makedirs(args.out_dir, exist_ok=True)
    table1(args.counts, os.path.join(args.out_dir, "tableS3_env_counts"), args.font, args.font_pt)
    table2(args.direction, os.path.join(args.out_dir, "tableS4_env_direction"), args.font,
           args.font_pt)


if __name__ == "__main__":
    main()
