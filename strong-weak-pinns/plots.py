"""The post's three figures, generated only from saved measurements."""
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/strong-weak-pinns-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, PercentFormatter
import numpy as np

STRONG, WEAK = "#2a78d6", "#eb6834"
STYLE = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": .8,
         "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
         "savefig.dpi": 180}


def plot(results, figures):
    metadata = json.loads((results / "metadata.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("Training is incomplete; finish the run before plotting.")
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    with np.load(results / "histories.npz") as archive:
        data = dict(archive)
    with np.load(results / "conditioning.npz") as archive:
        conditions = dict(archive)
    seeds, problems, counts = metadata["seeds"], metadata["problems"], metadata["counts"]
    reference = metadata["comparison_tests"]
    labels = (r"Strong PINN, $\mathcal{L}_s$", rf"Weak PINN, $\mathcal{{L}}_w$ ($m={reference}$)")
    width = 3 * len(problems)

    def runs(problem, method):
        """Array (seed, record, column) with columns step, own loss, common error."""
        return np.stack([data[f"{problem['key']}_{method}_seed{s}"] for s in seeds])

    def band(ax, x, y, color, label):
        ax.plot(x, np.median(y, axis=0), color=color, label=label)
        ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=.1, linewidth=0)

    def percent(ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_minor_formatter(NullFormatter())

    def finish(fig, name, top):
        handles, names = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.))
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(figures / name)
        plt.close(fig)

    # 1. One column per problem: each method's own loss above the common error.
    fig, axes = plt.subplots(2, len(problems), figsize=(width, 6.6), sharex=True, sharey="row")
    for col, problem in enumerate(problems):
        for method, color, label in zip(("strong", f"weak-{reference}"), (STRONG, WEAK), labels):
            history = runs(problem, method)
            for row in (0, 1):
                band(axes[row, col], history[0, :, 0], history[:, :, row + 1], color, label)
        axes[0, col].set_title(problem["title"])
        axes[1, col].set_xlabel("Adam step")
    for ax in axes.flat:
        ax.set_yscale("log")
    percent(axes[1, 0])
    axes[0, 0].set_ylabel(r"Own loss $\mathcal{L}_s$ or $\mathcal{L}_w$")
    axes[1, 0].set_ylabel(r"Common error $\mathcal{L}$")
    finish(fig, "training.png", .92)

    # 2. Condition numbers of the loss Hessians at u* in sine coefficients.
    fig, axes = plt.subplots(1, len(problems), figsize=(width, 3.9), sharex=True, sharey=True)
    for ax, problem in zip(axes, problems):
        for j, color, label in zip((0, 1), (STRONG, WEAK), (r"Strong loss $\mathcal{L}_s$", r"Weak loss $\mathcal{L}_w$")):
            ax.loglog(conditions["N"], conditions[problem["key"]][:, j], "o-", color=color, label=label)
        ax.set(title=problem["title"], xlabel="Number of modes $N$")
        ax.set_xticks(conditions["N"], labels=conditions["N"])
        ax.xaxis.set_minor_locator(NullLocator())
    for j, text in ((0, "$N^4$"), (1, "$N^2$")):
        axes[0].annotate(text, (conditions["N"][-1], conditions["poisson"][-1, j]),
                         xytext=(-24, -3), textcoords="offset points", color="#444444")
    axes[0].set_ylabel(r"Condition number $\kappa(H)$")
    finish(fig, "conditioning.png", .86)

    # 3. Final common error of the weak PINN against its number of tests.
    fig, axes = plt.subplots(1, len(problems), figsize=(width, 3.9), sharex=True, sharey=True)
    for ax, problem in zip(axes, problems):
        strong = runs(problem, "strong")[:, -1, 2]
        ax.axhline(np.median(strong), color=STRONG, ls="--", label="Strong PINN")
        ax.axhspan(strong.min(), strong.max(), color=STRONG, alpha=.1, linewidth=0)
        weak = np.array([runs(problem, f"weak-{m}")[:, -1, 2] for m in counts])
        median = np.median(weak, axis=1)
        ax.errorbar(counts, median, yerr=[median - weak.min(1), weak.max(1) - median],
                    color=WEAK, marker="o", capsize=3, label="Weak PINN")
        ax.set(xscale="log", yscale="log", title=problem["title"], xlabel="Number of test functions $m$")
        ax.set_xticks(counts, labels=counts)
        ax.xaxis.set_minor_locator(NullLocator())
    percent(axes[0])
    axes[0].set_ylabel(r"Final common error $\mathcal{L}$")
    finish(fig, "projection.png", .86)
