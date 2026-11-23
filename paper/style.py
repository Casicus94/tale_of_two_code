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


def save_fig(fig, name, line_art=False):
    """Write png (300 dpi; 600 for pure line art) and pdf into Paper/figures/."""
    fig.savefig(FIG / f"{name}.png", dpi=600 if line_art else 300)
    fig.savefig(FIG / f"{name}.pdf")
    if not NOTEBOOK:
        plt.close(fig)
    print("saved", FIG / f"{name}.png")


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
