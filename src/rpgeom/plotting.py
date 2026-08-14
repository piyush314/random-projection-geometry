"""
Economist-style plotting module for academic papers.

Usage:
    from economist_style import setup_style, line_chart, bar_chart, scatter_plot, save_figure

    setup_style()
    fig, ax = line_chart(x, [y1, y2], ['Series 1', 'Series 2'],
                         title='Main Finding in Title',
                         subtitle='Source: Dataset Name')
    save_figure(fig, 'figures/plots/my_plot')
"""

import matplotlib.pyplot as plt
import numpy as np

# Color palette
COLORS = {
    "red": "#E3120B",
    "blue": "#006BA2",
    "teal": "#00A5A5",
    "gold": "#F4A100",
    "purple": "#6F2DA8",
    "green": "#00843D",
    "dark_gray": "#3D3D3D",
    "medium_gray": "#767676",
    "light_gray": "#D0D0D0",
}

COLOR_SEQUENCE = [
    COLORS["red"],
    COLORS["blue"],
    COLORS["teal"],
    COLORS["gold"],
    COLORS["purple"],
    COLORS["green"],
]

COLUMN_WIDTHS = {
    "single": 3.5,
    "double": 7.0,
    "thesis": 6.0,
    "beamer": 4.5,
}


def setup_style():
    """Apply Economist-style defaults globally."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 20,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["dark_gray"],
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": COLORS["dark_gray"],
            "ytick.color": COLORS["dark_gray"],
            "grid.color": COLORS["light_gray"],
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def get_figsize(width="single", aspect_ratio=0.618):
    """Get figure size for LaTeX integration."""
    w = COLUMN_WIDTHS.get(width, 3.5)
    return (w, w * aspect_ratio)


def add_subtitle(ax, subtitle, y_offset=1.02):
    """Add subtitle below title."""
    ax.text(
        0,
        y_offset,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["medium_gray"],
        ha="left",
        va="bottom",
    )


def apply_grid(ax, direction="horizontal"):
    """Apply minimal gridlines."""
    if direction in ("horizontal", "both"):
        ax.yaxis.grid(True, linestyle="-", alpha=0.7, color=COLORS["light_gray"])
    if direction in ("vertical", "both"):
        ax.xaxis.grid(True, linestyle="-", alpha=0.7, color=COLORS["light_gray"])


def line_chart(
    x, y_series, labels, title, subtitle="", ylabel="", xlabel="", figsize=None, width="single"
):
    """Create Economist-style line chart."""
    if figsize is None:
        figsize = get_figsize(width)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (y, label) in enumerate(zip(y_series, labels)):
        ax.plot(x, y, color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], linewidth=2, label=label)

    ax.set_title(title, loc="left", fontweight="bold")
    if subtitle:
        add_subtitle(ax, subtitle)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

    apply_grid(ax, "horizontal")

    if len(y_series) > 1:
        ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    return fig, ax


def bar_chart(
    categories,
    values,
    title,
    subtitle="",
    ylabel="",
    horizontal=False,
    figsize=None,
    width="single",
    highlight_idx=None,
    bar_color="blue",
):
    """Create Economist-style bar chart."""
    if figsize is None:
        figsize = get_figsize(width)

    fig, ax = plt.subplots(figsize=figsize)

    default_color = COLORS.get(bar_color, COLORS["blue"])
    colors = [default_color] * len(values)
    if highlight_idx is not None:
        colors[highlight_idx] = COLORS["red"]

    if horizontal:
        ax.barh(categories, values, color=colors, height=0.6)
        if ylabel:
            ax.set_xlabel(ylabel)
        apply_grid(ax, "vertical")
    else:
        ax.bar(categories, values, color=colors, width=0.6)
        if ylabel:
            ax.set_ylabel(ylabel)
        apply_grid(ax, "horizontal")

    ax.set_title(title, loc="left", fontweight="bold")
    if subtitle:
        add_subtitle(ax, subtitle)

    plt.tight_layout()
    return fig, ax


def scatter_plot(
    x,
    y,
    title,
    subtitle="",
    xlabel="",
    ylabel="",
    sizes=None,
    colors=None,
    labels=None,
    show_trend=False,
    figsize=None,
    width="single",
):
    """Create Economist-style scatter plot."""
    if figsize is None:
        figsize = get_figsize(width, aspect_ratio=0.8)

    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = COLORS["blue"]
    if sizes is None:
        sizes = 50

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, edgecolors="white", linewidth=0.5)

    if show_trend:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "--", color=COLORS["medium_gray"], linewidth=1)

    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")

    ax.set_title(title, loc="left", fontweight="bold")
    if subtitle:
        add_subtitle(ax, subtitle)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    apply_grid(ax, "both")
    plt.tight_layout()
    return fig, ax


def save_figure(fig, filename, formats=None):
    """Save figure in formats suitable for LaTeX."""
    if formats is None:
        formats = ["pdf", "png"]
    for fmt in formats:
        fig.savefig(
            f"{filename}.{fmt}", bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none"
        )
    plt.close(fig)


if __name__ == "__main__":
    # Demo
    setup_style()

    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 50)

    fig, ax = line_chart(
        x,
        [y1, y2],
        ["Sine", "Cosine"],
        title="Trigonometric functions show periodic behavior",
        subtitle="Source: Mathematical simulation",
    )
    save_figure(fig, "demo_line")
    print("Demo saved to demo_line.pdf")
