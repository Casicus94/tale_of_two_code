"""Shared figure style for the paper (matches the slide deck palette)."""
from pathlib import Path

import sys
import matplotlib

NOTEBOOK = "ipykernel" in sys.modules      # in Jupyter: use the inline backend and keep figures open so they display
if not NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "Paper" / "figures"
DATA_OUT = FIG / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

PM25 = "#D98324"   # orange (main-text PM2.5 identity colour)
PM25_BLUE = "#3D74B6"   # medium/steel blue used instead of PM25 in Fig 3 and every SI figure (2026-09-23, user request)
O3 = "#2A8C8A"     # teal
NAVY = "#16283C"   # text and axes
GREY = "#A9B6C2"   # baselines and literature
LIGHT = "#E8EDF1"
RED = "#B23A3A"    # caveats / negative

plt.rcParams.update({
    # Type 3 (matplotlib's PDF default) embeds each glyph as a custom-encoded mini vector program;
    # several PDF viewers mis-render specific glyphs from that encoding (seen as missing "i"/"l"
    # letters in exported figures) even though the same figure rasterizes fine as PNG. TrueType (42)
    # embeds real glyph outlines instead and avoids this. See MASTER.md, 2026-09-24.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": False,
})


import re as _re
from matplotlib.text import Text as _Text

# Single letters that are symbols on our figures (n days, r correlation, t time, w vertical velocity, u wind ...):
# a text starting with one of them followed by a non-letter keeps its lower case.
_SYMBOL_START = _re.compile(r"^[nrtwuxykp](?![A-Za-z])")


def _elementa_fix(s):
    if not isinstance(s, str) or not s:
        return s
    m = _re.fullmatch(r"\(([a-h])\)", s.strip())
    if m:
        return m.group(1)
    s = s.replace("Lead eq.", "lead equation").replace("Eq. ", "Equation ")
    if s[0].islower() and not _SYMBOL_START.match(s):
        s = s[0].upper() + s[1:]
    return s


def _elementa_text(fig):
    """Apply the Elementa style sheet to every text on the figure (2026-11-23):
    panel letters without parentheses, sentence case for labels and phrases, and
    'Equation' spelled out instead of 'Eq.'."""
    from matplotlib.ticker import FixedFormatter, FuncFormatter
    from matplotlib.category import StrCategoryFormatter
    for ax in fig.axes:                      # text tick labels are regenerated at draw time from the formatter
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, (FixedFormatter, FuncFormatter, StrCategoryFormatter)) and not getattr(fmt, "_elementa", False):
                new = FuncFormatter(lambda x, pos=None, _f=fmt: _elementa_fix(_f(x, pos)))
                new._elementa = True
                axis.set_major_formatter(new)
    for t in fig.findobj(_Text):
        s = t.get_text()
        new = _elementa_fix(s)
        if new != s:
            t.set_text(new)


def save_fig(fig, name, line_art=False):
    """Write png (300 dpi; 600 for pure line art) and pdf into Paper/figures/."""
    _elementa_text(fig)
    fig.savefig(FIG / f"{name}.png", dpi=600 if line_art else 300)
    fig.savefig(FIG / f"{name}.pdf")
    if not NOTEBOOK:
        plt.close(fig)
    print("saved", FIG / f"{name}.png")


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
