# IMPORTS

    # PYTHON LIBRARIES

import pickle as pkl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

    # OTHER CODE FILES

import sim_core_v2
import sim_sweep_v2
from sim_sweep_v2 import *

# STATIC MODEL PARAMETERS
MONTH_LEN = 30.44      # days/month (~365/12)
N_MONTHS = 36           # 36-month observation window
DEFAULT_T = 365 * 3

# VISUALIZATION FUNCTIONS

def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"

METRIC_LABELS = {
    "# Ever Homeless": "Ever homeless (count)",
    "# Consecutive": "Ever consecutively chronic (count)",
    "# Episodic": "Ever episodically chronic (count)",
    "Episodic as % Chronic": "Episodic (% of all chronic)",
}

def plot_heatmap_grid(summary_df, x_param, y_param, pu, metrics=None, agg="mean",
                       fixed_params_note=None, save_path=None, cmap="plasma"):
    set_style()
    if metrics is None:
        metrics = list(METRIC_LABELS.keys())
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    x_vals = sorted(summary_df[x_param].unique())
    y_vals = sorted(summary_df[y_param].unique(), reverse = True)
    for i, metric in enumerate(metrics):
        pivot = summary_df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc=agg)
        pivot = pivot.reindex(index=y_vals, columns=x_vals)
        sns.heatmap(pivot, ax=axes[i], annot=True, fmt='.2f', cmap=cmap,
                    annot_kws={"size": 11}, cbar_kws={"label": metric})
        axes[i].set_title(METRIC_LABELS.get(metric, metric), fontsize=26)
        axes[i].set_xlabel(x_param); axes[i].set_ylabel(y_param)
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")
    title = f"Chronicity outcomes averaged across simulations ({agg})\nP undetected = " + str(pu)
    if fixed_params_note:
        title += f"\n{fixed_params_note}"
    fig.suptitle(title, y=1.02, fontsize=48)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_undetected_heatmap_grid(summary_df, x_param="P homeless", y_param="P exit",
                                 facet_param="P undetected", metric="Frac Undetected Chronic",
                                 agg="mean", cmap="rocket_r", annot=True, fmt=".2f",
                                 share_scale=True, facet_order=None, panel_size=4.5,
                                 fixed_params_note=None, save_path=None):
    """
    Facet a single metric across one row of heatmap panels, one panel per
    value of `facet_param` (e.g. P undetected), with `x_param`/`y_param`
    (e.g. P homeless / P exit) forming the axes of each heatmap.

    share_scale=True puts every panel on the same color scale (min/max
    taken over the whole summary_df) so panels are directly comparable;
    set False to let each panel auto-scale to its own range instead.
    """
    set_style()

    facet_vals = facet_order or sorted(summary_df[facet_param].unique())
    x_vals = sorted(summary_df[x_param].unique())
    y_vals = sorted(summary_df[y_param].unique(), reverse=True)

    ncols = len(facet_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(panel_size * ncols + 1.5, panel_size + 0.5),
                             sharey=True)
    axes = np.atleast_1d(axes).flatten()

    vmin, vmax = (None, None)
    if share_scale:
        vmin, vmax = summary_df[metric].min(), summary_df[metric].max()

    for i, (ax, fval) in enumerate(zip(axes, facet_vals)):
        sub = summary_df.loc[summary_df[facet_param] == fval]
        pivot = sub.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc=agg)
        pivot = pivot.reindex(index=y_vals, columns=x_vals)
        sns.heatmap(pivot, ax=ax, annot=annot, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, annot_kws={"size": 10},
                    cbar=(i == ncols - 1),
                    cbar_kws={"label": METRIC_LABELS.get(metric, metric)} if i == ncols - 1 else None)
        ax.set_title(f"{facet_param} = {fval}", fontsize=16)
        ax.set_xlabel(x_param)
        if i == 0:
            ax.set_ylabel(y_param)
        else:
            ax.set_ylabel("")

    title = f"{METRIC_LABELS.get(metric, metric)}\nfaceted by {facet_param} (agg={agg})"
    if fixed_params_note:
        title += f"\n{fixed_params_note}"
    fig.suptitle(title, y=1.05, fontsize=20)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def plot_timeseries(timeseries, keys, title=None, save_path=None, logy=False):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Ever_homeless": "#1f77b4", "Consecutive_counts": "#2ca02c", "Episodic_counts": "#7f7f7f"}
    labels_used = set()
    for key in keys:
        if key not in timeseries:
            continue
        series = timeseries[key]
        for col, label in [("Ever_homeless", "Ever homeless"), ("Consecutive_counts", "Ever consecutive"),
                            ("Episodic_counts", "Ever episodic")]:
            lbl = label if label not in labels_used else None
            ax.plot(series["Times"], series[col], color=colors[col], alpha=0.5, label=lbl)
            labels_used.add(label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Timestep (days)"); ax.set_ylabel("Count")
    ax.set_title(title or "System-wide chronicity counts over time")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_timeseries_ax(timeseries, keys, ax, title=None, save_path=None, logy=False,
                        xlabel="", ylabel="", show_legend=False):
    set_style()
    colors = {"Ever_homeless": "#1f77b4", "Consecutive_counts": "#2ca02c", "Episodic_counts": "#7f7f7f"}
    plot_labels = {"Ever_homeless": "Ever-homeless", "Consecutive_counts": "Ever chronic",
                   "Episodic_counts": "Ever episodic"}
    labels_used = set()
    ax.tick_params(axis='both', labelsize=32)
    for key in keys:
        if key not in timeseries:
            continue
        series = timeseries[key]
        for col in ("Ever_homeless", "Consecutive_counts", "Episodic_counts"):
            label = plot_labels[col]
            lbl = label if label not in labels_used else None
            ax.plot(series["Times"], series[col], color=colors[col], alpha=0.5, label=lbl)
            labels_used.add(label)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=52)
    ax.set_xlabel(xlabel, fontsize=64)
    ax.set_ylabel(ylabel, fontsize=64)
    if show_legend:
        ax.legend(loc="best", prop={"size": 42, "weight": "bold"}, labelcolor="black")
    return ax


def plot_undetected_stripplot(summary_df, x_param, hue_param, pu, save_path=None):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.stripplot(data=summary_df, x=x_param, y="Frac Undetected Chronic", hue=hue_param,
                  dodge=True, jitter=True, alpha=0.75, palette="Set1", ax=ax)
    ax.set_title(f"Undetected episodic chronicity, by {x_param} / {hue_param}\nP_u = {pu}")
    ax.set_ylabel("% Episodic Chronicity Undetected")
    ax.legend(title=hue_param, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig

# RUN MODEL TO GENERATE DATA

# if __name__ == "__main__":
#
#     P_h_example = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
#     P_e_example = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
#     P_u_example = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
#
#     summary_df, timeseries = run_sweep(
#         P_h_example, P_e_example, P_u_example,
#         numsims=10, size_pop=20000, T=DEFAULT_T,
#         record_timeseries=True, fix_known_bugs=True,
#         n_workers=None, base_seed=None,
#     )
#
#     save_sweep(summary_df, timeseries, out_dir="sweep_results", tag="pu_pe_ph")
    # print(summary_df.head())

# IMPORT MODEL DATA

    # summary_df, timeseries = load_sweep(out_dir="sweep_results", tag="pu_pe_ph")

summary_df, timeseries = load_sweep(out_dir="sweep_results", tag="pu_pe_ph")

# CONTROL FOR P_U

# summary_df = summary_df.loc[summary_df['P undetected'] == XXX]

# VISUALIZATIONS

    # HEAT MAPS

        # FOR VARYING P_H, P_E
# aggdata = pd.DataFrame(columns = [0.0025, 0.005, 0.0075, 0.01, 0.125])
# for pu in [0.0025, 0.005, 0.0075, 0.01, 0.0125]:
#     summary_df_pu = summary_df.loc[summary_df['P undetected'] == pu]
#     fig1 = plot_heatmap_grid(summary_df_pu, x_param="P homeless", y_param="P exit", pu=pu,
#                              save_path="images/heatmaps_pe_ph_"+str(pu)+".png")


        # FOR VARYING P_U

# METRIC_LABELS.update({
#     "Frac Undetected Chronic": "Undetected chronic (frac. of chronic pop.)",
# })
# fig = plot_undetected_heatmap_grid(
#     summary_df,
#     x_param="P homeless", y_param="P exit", facet_param="P undetected",
#     metric="Frac Undetected Chronic", agg="mean", cmap="rocket_r",
#     save_path="images/undetected_heatmap_grid.png",
# )

        # TIMESERIES MEGA-PANEL COMPARING CHRONICITY COUNTS OVER TIME WITH VARYING P_H, P_E

# with open("sweep_results/pu_pe_ph_timeseries.pkl", "rb") as f:
#     timeseries = pkl.load(f)     # dict_keys formatted (sim, p_h, p_e, p_u)
#
# p_u_values = sorted({k[3] for k in timeseries.keys()})
#
# for p_u in p_u_values:
#     fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(90, 60))
#
#     for y in range(25, 126, 25):
#         for z in range(125, 24, -25):
#             thisrow = int((z / 25) - 1)
#             thiscol = int((y / 25) - 1)
#             thisax = axes[thisrow, thiscol]
#             this_key = (0, y / 10000, z / 10000, p_u)
#             matching_keys = [k for k in timeseries if k[1] == this_key[1] and k[2] == this_key[2] and k[3] == this_key[3]]
#             is_legend_ax = (thisrow == 4 and thiscol == 4)
#
#             if y == 25 and thisrow != 4:
#                 plot_timeseries_ax(timeseries, keys=matching_keys, ax=thisax,
#                                     title=f"p_h={this_key[1]}, p_e={this_key[2]}",
#                                     ylabel="P_e = " + str(z / 10000),
#                                     show_legend=is_legend_ax)
#             elif thisrow == 4 and y == 25:
#                 plot_timeseries_ax(timeseries, keys=matching_keys, ax=thisax,
#                                     title=f"p_h={this_key[1]}, p_e={this_key[2]}",
#                                     xlabel="P_h = " + str(y / 10000),
#                                     ylabel="P_e = " + str(z / 10000),
#                                     show_legend=is_legend_ax)
#             elif thisrow == 4 and y != 25:
#                 plot_timeseries_ax(timeseries, keys=matching_keys, ax=thisax,
#                                     title=f"p_h={this_key[1]}, p_e={this_key[2]}",
#                                     xlabel="P_h = " + str(y / 10000),
#                                     show_legend=is_legend_ax)
#             else:
#                 plot_timeseries_ax(timeseries, keys=matching_keys, ax=thisax,
#                                     title=f"p_h={this_key[1]}, p_e={this_key[2]}",
#                                     show_legend=is_legend_ax)
#
#     fig.subplots_adjust(left=0.06, bottom=0.06)
#     fig.text(0.015, 0.5, "Count", rotation="vertical", fontsize=64, fontweight="bold", ha="center", va="center")
#     fig.text(0.5, 0.015, "Timestep (Days)", rotation="horizontal", fontsize=64, fontweight="bold", ha="center", va="center")
#
#     fig.suptitle(f'Effects of P Homeless, P Exit on\nSystem-wide Chronicity Counts over Time (p_u = {p_u})',
#                  fontsize=100, fontname="Georgia", fontweight="bold")
#     fig.savefig(f'images/timeseries_panel_pu_{p_u}.jpg')
#     plt.close(fig)

    # STRIP PLOTS

for pu in [0.0025, 0.005, 0.0075, 0.01, 0.0125]:
    summary_df_pu = summary_df.loc[summary_df['P undetected'] == pu]
    fig3 = plot_undetected_stripplot(summary_df_pu, x_param="P homeless",pu = pu, hue_param="P exit",
                                     save_path=f"images/undetected_stripplot_v2_{pu}.png")