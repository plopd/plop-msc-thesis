#!/usr/bin/env python


def lineplot(
    ax,
    x,
    y,
    yerr,
    label,
    marker=None,
    show_legend=False,
    xscale={},
    ylim={},
    n_std=None,
):

    ax.plot(x, y, label=label, marker=marker)

    if n_std is not None:
        ax.fill_between(
            x, y + n_std * yerr, y - n_std * yerr, alpha=0.1,
        )
    if show_legend:
        ax.legend()

    ax.set_xscale(xscale.get("value"), basex=xscale.get("base"))

    ax.set_ylim(ylim.get("bottom", None), ylim.get("top", None))

    return ax
