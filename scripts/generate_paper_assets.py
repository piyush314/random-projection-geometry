"""Generate the manuscript's five figures from deterministic seeds."""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reproduction.paper.verify_claims import anisotropy_sweep, beta_arcsine_probability
from rpgeom.plotting import COLORS, setup_style

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "paper" / "figures")
args = parser.parse_args()

setup_style()
rng = np.random.default_rng(42)
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)

def polish(ax):
    ax.grid(axis='y', alpha=0.5); ax.grid(axis='x', visible=False)


# Figures 2--5 appear two to a row in the manuscript. Generate them at their
# final panel width so LaTeX does not shrink text designed for a wide figure.
# With the manuscript geometry these settings yield effective label, tick,
# and legend sizes of approximately 10, 9, and 9 pt.
COMPACT_FIGSIZE = (3.45, 2.75)
COMPACT_TITLE_SIZE = 13.5
COMPACT_LABEL_SIZE = 12.5
COMPACT_TICK_SIZE = 11
COMPACT_LEGEND_SIZE = 11


def polish_compact(fig, ax):
    polish(ax)
    ax.title.set_fontsize(COMPACT_TITLE_SIZE)
    ax.xaxis.label.set_fontsize(COMPACT_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(COMPACT_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=COMPACT_TICK_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for label in legend.get_texts():
            label.set_fontsize(COMPACT_LEGEND_SIZE)
    fig.tight_layout(pad=0.35)

# ---------------- Figure 1: order-preservation probability -------------------
d = 200
ms = np.arange(2, 121, 2)
p_exact, p_mc = [], []
for m in ms:
    p_exact.append(beta_arcsine_probability(m, d))
for m in [5, 10, 20, 40, 60, 80, 100, 120]:
    N = 120000
    beta = rng.beta(m/2, (d-m)/2, N)
    rho = np.sqrt(beta)
    score = rng.standard_normal(N)
    projected = rho*score + np.sqrt(1-beta)*rng.standard_normal(N)
    p_mc.append((m, np.mean(score*projected > 0)))
fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE)
x = ms/d
ax.plot(x, p_exact, color=COLORS['red'], lw=2, label='exact law')
ax.plot(x, 0.5 + np.sqrt(x)/np.pi, color=COLORS['blue'], lw=1.4, ls='--',
        label='asymptotic')
mx, my = zip(*p_mc)
ax.scatter(np.array(mx)/d, my, color=COLORS['dark_gray'], s=22, zorder=5, label='Monte Carlo')
ax.axhline(0.5, color=COLORS['medium_gray'], lw=0.8)
ax.set_xlabel('compression ratio  $m/d$'); ax.set_ylabel('$p_{m,d}$')
ax.set_title('Pairwise ordering')
ax.legend(frameon=False, loc='lower right')
polish_compact(fig, ax)
fig.savefig(OUT / 'fig_order.pdf', bbox_inches='tight')

# ---------------- Figure 2: Laguerre spectrum ------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ratios = np.linspace(0.005, 0.5, 200)
d = 400
cols = [COLORS['red'], COLORS['blue'], COLORS['teal'], COLORS['gold']]
for k, c in zip(range(1, 5), cols):
    sk2 = []
    for rr in ratios:
        m = rr*d
        num = np.prod([m/2 + j for j in range(k)])
        den = np.prod([d/2 + j for j in range(k)])
        sk2.append(num/den)
    ax.plot(ratios, sk2, color=c, lw=2, label=f'$k={k}$')
    ax.plot(ratios, ratios**k, color=c, lw=1, ls=':')
ax.set_yscale('log'); ax.set_ylim(1e-8, 1.2)
ax.set_xlabel('compression ratio  $m/d$')
ax.set_ylabel(r'recoverable variance share  $s_k^2$')
ax.set_title('Laguerre grading of recoverable variance', fontsize=12)
ax.text(0, 1.005, r'Laguerre singular values $s_k^2=(m/2)_k/(d/2)_k$ (solid) versus $(m/d)^k$ (dotted), $d=400$',
        transform=ax.transAxes, fontsize=8, color=COLORS['medium_gray'])
ax.legend(frameon=False, fontsize=8, loc='lower right')
polish(ax)
fig.savefig(OUT / 'fig_laguerre.pdf', bbox_inches='tight')

# ---------------- Figure 3: harmonic law for iid transforms ------------------
d = 200
ms = np.arange(10, 801, 10)
fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE)
ax.plot(ms/d, np.sqrt(np.minimum(ms/d, 1)), color=COLORS['blue'], lw=2,
        label='Haar projection')
ax.plot(ms/d, np.sqrt(ms/(ms+d+2)), color=COLORS['red'], lw=2,
        label='i.i.d. Gaussian')
mc = []
for m in [25, 50, 100, 200, 400, 800]:
    N = 150000
    Dl = rng.chisquare(d, N)
    Al = Dl*rng.chisquare(m, N)/m
    mc.append((m, np.corrcoef(Dl, Al)[0, 1]))
mx, my = zip(*mc)
ax.scatter(np.array(mx)/d, my, color=COLORS['dark_gray'], s=22, zorder=5, label='Monte Carlo (i.i.d.)')
ax.set_xlabel('$m/d$'); ax.set_ylabel(r'$\mathrm{Corr}(D,\widetilde D)$')
ax.set_title('Map correlation')
ax.legend(frameon=False, loc='lower right')
polish_compact(fig, ax)
fig.savefig(OUT / 'fig_harmonic.pdf', bbox_inches='tight')

# ---------------- Figure 4: information-geometry contraction -----------------
d = 60
ms = np.arange(2, 59, 2)
fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE)
ax.plot(ms/d, ms/d, color=COLORS['blue'], lw=2, label='mean / log scale')
shape = (ms-1)*(ms+2)/((d-1)*(d+2))
ax.plot(ms/d, shape, color=COLORS['red'], lw=2,
        label='shape (exact)')
mc = []
H = rng.standard_normal((d, d)); H = (H+H.T)/2; H -= np.trace(H)/d*np.eye(d)
for m in [4, 8, 16, 24, 32, 40, 48, 56]:
    r = []
    for _ in range(1500):
        Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
        C = Q.T @ H @ Q
        C0 = C - np.trace(C)/m*np.eye(m)
        r.append((C0**2).sum()/(H**2).sum())
    mc.append((m, np.mean(r)))
mx, my = zip(*mc)
ax.scatter(np.array(mx)/d, my, color=COLORS['dark_gray'], s=22, zorder=5,
           label='shape (Monte Carlo)')
ax.set_xlabel('$m/d$'); ax.set_ylabel('retained fraction')
ax.set_title('Local information')
ax.legend(frameon=False, loc='upper left')
polish_compact(fig, ax)
fig.savefig(OUT / 'fig_infogeo.pdf', bbox_inches='tight')

# ---------------- Figure 5: anisotropic polynomial CCA -----------------------
rows = anisotropy_sweep()
lam = np.array([row['lambda_ratio'] for row in rows])
fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE)
ax.plot(lam, [row['sqrt_alpha_1'] for row in rows], color=COLORS['dark_gray'],
        lw=1.6, ls='--', label=r'distance value $\sqrt{\alpha_1}$')
for degree, color in zip([2, 3, 4], [COLORS['blue'], COLORS['teal'], COLORS['red']]):
    ax.plot(lam, [row[f'cca_degree_{degree}'] for row in rows], color=color,
            lw=2, label=fr'degree $\leq {degree}$')
ax.axvline(2, color=COLORS['medium_gray'], lw=0.8, ls=':')
ax.set_xscale('log')
ax.set_xlim(1, 10)
ax.set_xticks([1, 2, 5, 10])
ax.set_xticklabels(['1', '2', '5', '10'])
ax.minorticks_off()
ax.set_ylim(0.69, 1.005)
ax.set_xlabel(r'eigenvalue ratio  $\lambda_1/\lambda_2$')
ax.set_ylabel('correlation lower bound')
ax.set_title('Anisotropic bounds')
ax.legend(frameon=False, loc='lower right')
polish_compact(fig, ax)
fig.savefig(OUT / 'fig_anisotropy.pdf', bbox_inches='tight')
print(f'wrote five figures to {OUT}')
