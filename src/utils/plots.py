import math
from typing import Union

import matplotlib.pyplot as plt

DEFAULT_COL_WIDTH = 7
DEFAULT_ROW_WIDTH = 6


def get_simple_axis(width: int = 14, height: int = 6) -> plt.Axes:
    _, ax = plt.subplots(figsize=(width, height))
    return ax


def rotate_xticklabels(axes, angle=45):
    axes = _make_iter(axes)
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(angle)


def make_plot(n_plots,
              n_cols=2,
              row_height=DEFAULT_ROW_WIDTH,
              col_width=DEFAULT_COL_WIDTH,
              sharex='all',
              sharey='all',
              ret_fig=False
              ) -> Union[plt.Axes, tuple[plt.Figure, plt.Axes]]:
    """Provides axes for grid for given number of plots
    :param n_plots: int, number of plots
    :param n_cols: int, number of plots in single row
    :param row_height: float, size of single row
    :param col_width: float, size of single column
    :param sharex: bool or str, sharex parameter passed to plt.subplots
    :param sharey: bool or str, sharey parameter passed to plt.subplots
    :param ret_fig: Bool, whether return figure object
    :return: List[matplotlib.axes._base._AxesBase], list of ready to use axes
    """
    assert isinstance(n_plots, int)
    assert isinstance(n_cols, int)

    n_cols = min(n_plots, n_cols)
    n_rows = math.ceil(n_plots / n_cols)
    fig_size = n_cols * col_width, n_rows * row_height
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=sharex, sharey=sharey)

    if empty_plots := ((n_cols * n_rows) - n_plots):
        for ax in axes.flatten()[-empty_plots:]:
            ax.set_axis_off()

    if ret_fig:
        return fig, axes
    return axes


def _make_iter(obj):
    try:
        iter(obj)
    except TypeError:
        obj = [obj]
    return obj
