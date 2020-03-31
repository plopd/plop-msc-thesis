#!/usr/bin/env python


def lineplot(
    ax,
    x,
    y,
    yerr,
    label,
    marker=None,
    color=None,
    show_legend=False,
    xticks=None,
    xlim={},
    xscale={},
    yticks=None,
    ylim={},
    n_std=None,
):

    ax.plot(x, y, label=label, marker=marker, color=color)

    if n_std is not None:
        ax.fill_between(x, y + n_std * yerr, y - n_std * yerr, alpha=0.1, color=color)
    if show_legend:
        ax.legend()

    ax.set_xscale(xscale.get("value"), basex=xscale.get("base"))
    ax.set_xlim(xlim.get("bottom", None), xlim.get("top", None))

    ax.set_ylim(ylim.get("bottom", None), ylim.get("top", None))

    if yticks is not None:
        ax.set_yticks(yticks.get("values"))
        ax.set_yticklabels(yticks.get("labels"))

    if xticks is not None:
        ax.set_xticks(xticks.get("values"))
        ax.set_xticklabels(xticks.get("labels"))

    return ax
