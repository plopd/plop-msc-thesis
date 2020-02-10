def plot_CI(ax, xs, ys1, ys2, color):
    for num_se in [1.0, 2.0, 2.5]:
        ax.fill_between(
            xs, ys1 + num_se * ys2, ys1 - num_se * ys2, color=color, alpha=0.15
        )

    return ax
