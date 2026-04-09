from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

PALETTE = {
    'tls': '#0F766E',
    'sentiment': '#C2410C',
    'universe': '#475569',
    'energy': '#D97706',
    'industrials': '#0F766E',
    'utilities': '#1D4ED8',
    'finbert': '#1D4ED8',
    'lexicon': '#0F766E',
    'vader': '#D97706',
    'negative': '#B91C1C',
    'positive': '#0F766E',
    'neutral': '#94A3B8',
    'light': '#E2E8F0',
    'text': '#0F172A',
    'grid': '#CBD5E1',
    'background': '#FFFFFF',
}

CONTROL_LABELS = {
    'none': 'None',
    'beta_mkt': 'Market beta',
    'co2_assets': 'CO2/assets',
    'beta_mkt+co2_assets': 'Market + CO2/assets',
}

HORIZON_LABELS = {'3m': '3M', '6m': '6M', '1y': '1Y', '2y': '2Y', '5y': '5Y'}


def apply_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'figure.facecolor': PALETTE['background'],
            'axes.facecolor': PALETTE['background'],
            'savefig.facecolor': PALETTE['background'],
            'axes.edgecolor': PALETTE['grid'],
            'axes.labelcolor': PALETTE['text'],
            'axes.titlecolor': PALETTE['text'],
            'xtick.color': PALETTE['text'],
            'ytick.color': PALETTE['text'],
            'text.color': PALETTE['text'],
            'font.family': 'serif',
            'font.serif': ['STIX Two Text', 'Times New Roman', 'Georgia', 'DejaVu Serif', 'STIXGeneral'],
            'font.size': 11,
            'axes.titlesize': 15,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 12,
            'axes.labelweight': 'regular',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.title_fontsize': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.color': PALETTE['grid'],
            'grid.linewidth': 0.8,
            'grid.alpha': 0.6,
            'axes.grid': True,
        }
    )


def thesis_diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        'thesis_diverging',
        [PALETTE['negative'], '#F8FAFC', PALETTE['tls']],
    )


def thesis_sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        'thesis_sequential',
        ['#F8FAFC', '#CBD5E1', '#64748B', '#0F172A'],
    )


def style_axes(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.55)
    ax.grid(False, axis='x')
    ax.tick_params(axis='both', which='both', length=0)


def percent_formatter(decimals: int = 0):
    fmt = f'{{x:.{decimals}%}}'
    return FuncFormatter(lambda x, pos: fmt.format(x=x))


def decimal_formatter(decimals: int = 2):
    fmt = f'{{x:.{decimals}f}}'
    return FuncFormatter(lambda x, pos: fmt.format(x=x))


def tidy_legend(ax, *, ncol: int = 1, loc: str = 'best', title: str | None = None) -> None:
    leg = ax.legend(frameon=True, facecolor='white', edgecolor=PALETTE['grid'], ncol=ncol, loc=loc, title=title)
    if leg:
        leg.get_frame().set_alpha(0.95)


def save_figure(fig, path: Path, *, dpi: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=PALETTE['background'])
    plt.close(fig)
