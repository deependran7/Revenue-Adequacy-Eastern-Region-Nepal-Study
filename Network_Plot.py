import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

BOUNDARY_PATH = 'Data/Geo_Data_Folder/Geo_Files/eastern_boundary.shp'

_LOAD_DEFAULT = object()   # sentinel: load the boundary lazily, not at import time


def load_eastern_boundary(path=BOUNDARY_PATH):
    return gpd.read_file(path)


def _normalise_outages(branch_outages):
    """Accept None, a str, a list/tuple/set, a pandas Index/Series or a numpy array,
    and return a plain list of line names. ('Line', name) tuples are reduced to name."""
    if branch_outages is None:
        return []
    if isinstance(branch_outages, str):
        return [branch_outages]
    try:
        items = list(branch_outages)
    except TypeError:
        return [branch_outages]
    names = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "Line":
            item = item[1]
        names.append(item)
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Flow computation helper
# ──────────────────────────────────────────────────────────────────────────────
def get_line_flows(network, snapshot, branch_outages=None):
    """
    Return a Series of line active-power flows (MW, indexed by line name).

    - No outages  -> base-case flows from network.lines_t.p0 (as usual).
    - One outage  -> post-contingency flows from network.lpf_contingency().
    - Several     -> simultaneous outage: lines are removed from a copy of the
                     network and a linear power flow is re-run.
    Outaged lines always carry 0 MW.
    """
    outages = _normalise_outages(branch_outages)

    # ---- base case ----------------------------------------------------------
    if not outages:
        if not network.lines_t.p0.empty and snapshot in network.lines_t.p0.index:
            return network.lines_t.p0.loc[snapshot].astype(float)
        # no stored flows for this snapshot: solve an LPF on a copy
        try:
            n_c = network.copy(snapshots=[snapshot])
            n_c.lpf(snapshot)
            return n_c.lines_t.p0.loc[snapshot].reindex(network.lines.index).astype(float)
        except Exception as e:
            print(f"[get_line_flows] base-case LPF failed: {e}")
            return None

    # lpf_contingency needs a base-case lpf for the snapshot
    if network.lines_t.p0.empty or snapshot not in network.lines_t.p0.index:
        network.lpf(snapshot)

    # ---- single outage: use lpf_contingency ---------------------------------
    if len(outages) == 1:
        outage = outages[0]
        try:
            res = network.lpf_contingency(
                snapshots=snapshot,
                branch_outages=[("Line", outage)],
            )
            # res: index = (component, name) branches; columns = ["base", outages...]
            col = ("Line", outage)
            if col not in res.columns:
                col = outage if outage in res.columns else res.columns[-1]
            flows = res[col]
            if isinstance(flows.index, pd.MultiIndex):
                flows = flows.xs("Line", level=0)
            flows = flows.reindex(network.lines.index).astype(float)
            flows.loc[outage] = 0.0
            return flows
        except Exception as e:
            print(f"lpf_contingency failed ({e}); using re-solved LPF on a copy.")

    # ---- multiple outages (or fallback): remove lines and re-solve ----------
    n_c = network.copy(snapshots=[snapshot])
    for l in outages:
        if l in n_c.lines.index:
            n_c.remove("Line", l)
    n_c.lpf(snapshot)
    flows = n_c.lines_t.p0.loc[snapshot].reindex(network.lines.index).fillna(0.0)
    for l in outages:
        flows.loc[l] = 0.0
    return flows.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Main plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_network_flow(
    network,
    snapshot=None,
    branch_outages=None,           # list of line names, e.g. ['L12'] or ['L12', 'L30']
    show_flow_arrows=True,
    show_flow_labels=True,
    show_gen_labels=True,
    show_bus_labels=True,
    show_gen_flow_arrows=True,
    figsize=(10, 5),
    save_path=None,
    dpi=300,
    extra_annotations=None,
    ax=None,
    legend=True,
    eastern_boundary=_LOAD_DEFAULT,   # pass None to disable the boundary
    parallel_offset=0.012,            # sideways spacing of parallel circuits (data units)
    verbose=True,                     # print lines that get skipped
):
    """
    Plot a PyPSA network with line-flow and generator-flow arrows.
    If branch_outages is given, line flows are taken from lpf_contingency
    (post-contingency); otherwise base-case flows are shown.

    Fixes vs. the previous version:
      * Parallel circuits (same bus pair) no longer draw their arrows/labels on top
        of each other: they are spread sideways.
      * Zero-length lines (co-located buses) still get an arrow and label.
      * NaN flows are shown as 0.0 MW instead of crashing / disappearing.
      * Skipped lines are reported when verbose=True.
      * Axis limits include generator positions so nothing falls outside the view.
      * The boundary shapefile is loaded when the function is called, not at import.
    """

    if eastern_boundary is _LOAD_DEFAULT:
        eastern_boundary = load_eastern_boundary()

    voltage_colors = {
        11:  "#1d693a",
        33:  "#f90404",
        132: '#E24B4A',
        220: '#EF9F27',
        400: '#185FA5',
    }
    voltage_labels_map = {
        11: '11 kV', 33: '33 kV', 132: '132 kV', 220: '220 kV', 400: '400 kV',
    }
    OUTAGE_COLOR = '#8e44ad'   # distinct from the red voltage colours

    # ── Resolve snapshot ───────────────────────────────────────────────────────
    if snapshot is None:
        resolved_snapshot = network.snapshots[0]
    elif isinstance(snapshot, (int, np.integer)):
        resolved_snapshot = network.snapshots[snapshot]
    else:
        resolved_snapshot = snapshot

    outage_list = _normalise_outages(branch_outages)

    # ── Line flows: contingency if outages, else normal ────────────────────────
    p0 = None
    if show_flow_arrows:
        p0 = get_line_flows(network, resolved_snapshot, outage_list)

    gen_p = None
    if (show_gen_flow_arrows and not network.generators_t.p.empty
            and resolved_snapshot in network.generators_t.p.index):
        gen_p = network.generators_t.p.loc[resolved_snapshot]

    # ── Figure / axes ──────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('w')
    else:
        fig = ax.get_figure()
    ax.set_facecolor('w')

    # ── Base network (outaged lines highlighted) ───────────────────────────────
    line_colors = network.lines.v_nom.map(voltage_colors).fillna('#888888').copy()
    line_widths = pd.Series(2.0, index=network.lines.index)
    for l in outage_list:
        if l in line_colors.index:
            line_colors[l] = OUTAGE_COLOR
            line_widths[l] = 0.0      # hidden; dotted overlay is drawn below

    network.plot(
        ax=ax,
        line_colors=line_colors,
        line_widths=line_widths,
        bus_sizes=0.0001,
        bus_colors="#5D56C6",
    )

    # ── Line flow arrows / labels ──────────────────────────────────────────────
    if show_flow_arrows and p0 is not None:
        # group parallel circuits so they can be offset from each other
        pair_key = network.lines.apply(
            lambda r: tuple(sorted((r['bus0'], r['bus1']))), axis=1)
        group_size = pair_key.map(pair_key.value_counts())
        group_rank = pair_key.groupby(pair_key).cumcount()

        for line_name, row in network.lines.iterrows():
            if line_name not in p0.index:
                if verbose:
                    print(f"[flow plot] {line_name}: no flow value, skipped")
                continue
            bus0, bus1 = row['bus0'], row['bus1']
            if bus0 not in network.buses.index or bus1 not in network.buses.index:
                if verbose:
                    print(f"[flow plot] {line_name}: bus missing, skipped")
                continue

            flow = float(p0[line_name]) if pd.notna(p0[line_name]) else 0.0
            x0, y0 = network.buses.at[bus0, 'x'], network.buses.at[bus0, 'y']
            x1, y1 = network.buses.at[bus1, 'x'], network.buses.at[bus1, 'y']
            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)

            if length == 0:                       # co-located buses
                ux, uy = 1.0, 0.0
                span = 0.006
            else:
                ux, uy = dx / length, dy / length
                span = length * 0.08

            # perpendicular offset, computed from a canonical bus order so that
            # reversed-direction twins don't land on the same spot
            if bus0 <= bus1:
                sx0, sy0, sx1, sy1 = x0, y0, x1, y1
            else:
                sx0, sy0, sx1, sy1 = x1, y1, x0, y0
            sl = np.hypot(sx1 - sx0, sy1 - sy0) or 1.0
            px, py = -(sy1 - sy0) / sl, (sx1 - sx0) / sl
            off = (group_rank[line_name] - (group_size[line_name] - 1) / 2) * parallel_offset

            mx = (x0 + x1) / 2 + px * off
            my = (y0 + y1) / 2 + py * off

            # Outaged line: dotted, labelled, no arrow
            if line_name in outage_list:
                ax.plot([x0, x1], [y0, y1], color=OUTAGE_COLOR,
                        linestyle=':', linewidth=2.5, zorder=4)
                ax.annotate(
                    'OUTAGE\n(0.0 MW)',
                    xy=(mx, my), xytext=(mx + 0.005, my + 0.008),
                    fontsize=5.5, color=OUTAGE_COLOR, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='#f4ecf7',
                              edgecolor=OUTAGE_COLOR, alpha=0.9, linewidth=0.6),
                    zorder=8,
                )
                continue

            direction = 1 if flow >= 0 else -1
            ax.annotate(
                '',
                xy=(mx + direction * ux * span, my + direction * uy * span),
                xytext=(mx - direction * ux * span, my - direction * uy * span),
                arrowprops=dict(arrowstyle='-|>', color='black',
                                lw=1.4, mutation_scale=14),
                zorder=5,
            )

            if show_flow_labels:
                ax.annotate(
                    f'{abs(flow):.1f} MW',
                    xy=(mx, my), xytext=(mx + 0.005, my + 0.008),
                    fontsize=5.5, color='#1a1a2e', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffcc',
                              edgecolor='#999900', alpha=0.85, linewidth=0.5),
                    zorder=8,
                )

    # ── Generators ─────────────────────────────────────────────────────────────
    gen_xs, gen_ys = [], []
    for gen_idx, row in network.generators.iterrows():
        # NOTE: kept exactly as in the original code (generator 'lat' column
        # holds the x/longitude value and 'lon' holds y/latitude in this dataset)
        gen_x, gen_y = row.lat, row.lon
        bus_name = row['bus']
        if bus_name not in network.buses.index:
            continue
        gen_xs.append(gen_x)
        gen_ys.append(gen_y)

        bus_x = network.buses.at[bus_name, 'x']
        bus_y = network.buses.at[bus_name, 'y']

        dispatch = None
        if gen_p is not None and gen_idx in gen_p.index:
            dispatch = float(gen_p[gen_idx])
        active = dispatch is not None and dispatch > 0

        ax.plot([gen_x, bus_x], [gen_y, bus_y],
                color='#0077b6', linewidth=1.2, linestyle='--',
                alpha=0.85 if active else 0.25, zorder=3)
        ax.plot(gen_x, gen_y, 'o',
                color='#0077b6' if active else '#aaaaaa',
                markersize=3.5, zorder=4)

        if show_gen_flow_arrows and dispatch is not None:
            dx, dy = bus_x - gen_x, bus_y - gen_y
            length = np.hypot(dx, dy)
            if length > 0:
                ux, uy = dx / length, dy / length
                mx, my = (gen_x + bus_x) / 2, (gen_y + bus_y) / 2
                span = length * 0.10

                ax.annotate(
                    '',
                    xy=(mx + ux * span, my + uy * span),
                    xytext=(mx - ux * span, my - uy * span),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='#0077b6' if active else '#cccccc',
                        lw=1.6 if active else 0.8,
                        mutation_scale=13 if active else 8,
                        alpha=0.9 if active else 0.35,
                    ),
                    zorder=5,
                )
                if show_flow_labels and active:
                    ax.annotate(
                        f'{dispatch:.1f} MW',
                        xy=(mx + uy * 0.012, my - ux * 0.012),
                        fontsize=5.0, color='#004f80', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#d0eeff',
                                  edgecolor='#0077b6', alpha=0.85, linewidth=0.5),
                        zorder=8,
                    )

        if show_gen_labels:
            ax.annotate(
                str(gen_idx),
                xy=(gen_x, gen_y), xytext=(gen_x + 0.01, gen_y + 0.01),
                fontsize=6.5, color='#004f80', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#e6f3fb',
                          edgecolor='#0077b6', alpha=0.8, linewidth=0.5),
                zorder=7,
            )

    # ── Boundary ───────────────────────────────────────────────────────────────
    if eastern_boundary is not None:
        eastern_boundary.boundary.plot(
            ax=ax, color='#333333', linewidth=1.5, linestyle='--')

    # ── Bus labels ─────────────────────────────────────────────────────────────
    if show_bus_labels:
        for bus in network.buses.index:
            x = network.buses.at[bus, 'x']
            y = network.buses.at[bus, 'y']
            v = network.buses.at[bus, 'v_nom']
            tc = voltage_colors.get(v, '#222222')
            ax.annotate(
                str(bus).split('_')[0],
                xy=(x, y), xytext=(x + 0.012, y + 0.012),
                fontsize=7.5, fontweight='bold', color=tc,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=tc, alpha=0.75, linewidth=0.6),
                zorder=6,
            )

    if extra_annotations:
        for ann in extra_annotations:
            ax.annotate(
                ann['text'], xy=ann['xy'],
                fontsize=ann.get('fontsize', 10),
                fontweight=ann.get('fontweight', 'bold'),
            )

    # ── Legends ────────────────────────────────────────────────────────────────
    if legend:
        line_handles = [
            mlines.Line2D([], [], color=c, linewidth=2.5, label=voltage_labels_map[kv])
            for kv, c in voltage_colors.items()
            if kv in network.lines.v_nom.values
        ]
        if outage_list:
            line_handles.append(
                mlines.Line2D([], [], color=OUTAGE_COLOR, linewidth=2.5,
                              linestyle=':', label='Outaged Line (Tripped)'))

        legend_lines = ax.legend(
            handles=line_handles, title='Transmission Lines',
            title_fontsize=9, loc='upper left', fontsize=8.5,
            framealpha=0.92, edgecolor='#aaaaaa', fancybox=True)
        legend_lines.get_title().set_fontweight('bold')

        feature_handles = [
            mlines.Line2D([], [], color='#0077b6', marker='o', linestyle='None',
                          markersize=6, label='Hydro-Power Site'),
            mlines.Line2D([], [], color='#0077b6', linewidth=1.2, linestyle='--',
                          label='Generator → Bus Connector'),
        ]
        if show_flow_arrows:
            title_flow = 'Post-contingency flow' if outage_list else 'Line flow'
            feature_handles += [
                mlines.Line2D([], [], color='black', linewidth=1.4,
                              marker='>', markersize=7, label=f'{title_flow} direction'),
                mpatches.Patch(facecolor='#ffffcc', edgecolor='#999900',
                               label=f'{title_flow} (MW)'),
            ]
        legend_gen = ax.legend(
            handles=feature_handles, title='Features',
            title_fontsize=9, loc='upper right', fontsize=8.5,
            framealpha=0.92, edgecolor='#aaaaaa', fancybox=True)
        legend_gen.get_title().set_fontweight('bold')
        ax.add_artist(legend_lines)

    # ── Auto-fit extent (buses + generators + boundary) ────────────────────────
    all_x = list(network.buses['x']) + gen_xs
    all_y = list(network.buses['y']) + gen_ys
    if eastern_boundary is not None:
        b = eastern_boundary.total_bounds
        all_x += [b[0], b[2]]
        all_y += [b[1], b[3]]

    pad_x = (max(all_x) - min(all_x)) * 0.08
    pad_y = (max(all_y) - min(all_y)) * 0.08
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    title = f'Snapshot: {resolved_snapshot}'
    if outage_list:
        title += f'   |   N-1 outage: {", ".join(map(str, outage_list))}'
    # ax.set_title(title, fontsize=10)

    ax.set_xlabel('Longitude', fontsize=10, color='#444')
    ax.set_ylabel('Latitude', fontsize=10, color='#444')
    ax.tick_params(labelsize=8, colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    ax.grid(True, linestyle='--', linewidth=0.4, color='#cccccc', alpha=0.6)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def check_line_loading(network, snapshot=None, limit_pct=80.0, ax=None, title_prefix=""):
    """
    Compute line loading (after pf() or optimize()) and optionally plot a bar chart.

    Parameters
    ----------
    network      : pypsa.Network
    snapshot      : int, label, or None (None = worst-case snapshot across all lines)
    limit_pct     : float — warning threshold in % of s_nom
    ax            : matplotlib Axes or None (if provided, draws onto it; else creates new fig)
    title_prefix  : str — prepended to the plot title (e.g. scenario name)

    Returns
    -------
    df : DataFrame with loading details, sorted by loading_pct descending
    """
    if network.lines_t.p0.empty:
        raise ValueError("network.lines_t.p0 is empty — run network.pf() or network.optimize() first.")

    # ── Resolve snapshot ──────────────────────────────────────────────────────
    loading_all = network.lines_t.p0.abs().div(network.lines.s_nom) * 100
    if snapshot is None:
        snap = loading_all.max(axis=1).idxmax()
    elif isinstance(snapshot, int):
        snap = network.snapshots[snapshot]
    else:
        snap = snapshot

    p0 = network.lines_t.p0.loc[snap]
    loss = (p0 + network.lines_t.p1.loc[snap]).abs() if not network.lines_t.p1.empty else None

    # ── Build results table ───────────────────────────────────────────────────
    flow_mw = p0.reindex(network.lines.index).fillna(0.0)
    #s_nom = network.lines['s_nom']
    #print('s_nom',s_nom)
    s_nom = network.lines['s_nom_opt']
    #print('s_nom_opt',s_nom_opt)
    loading_pct = (flow_mw.abs() / s_nom * 100).fillna(0)

    status = pd.cut(
        loading_pct,
        bins=[-np.inf, limit_pct * 0.75, limit_pct, 100, np.inf],
        labels=['🟢 OK', '🟡 WARNING', '🟠 OVERLOADED', '🔴 CRITICAL'],
    )

    df = pd.DataFrame({
        'line': network.lines.index,
        'v_nom_kV': network.lines['v_nom'].astype(int),
        'bus0': network.lines['bus0'],
        'bus1': network.lines['bus1'],
        's_nom_MVA': s_nom.round(1),
        'flow_MW': flow_mw.round(2),
        'loading_pct': loading_pct.round(2),
        'headroom_MW': (s_nom * limit_pct / 100 - flow_mw.abs()).round(2),
        'status': status,
    }).sort_values('loading_pct', ascending=False)

    if loss is not None:
        df['loss_MW'] = loss.reindex(network.lines.index).fillna(0).round(3).values

    # ── Console summary (brief) ───────────────────────────────────────────────
    counts = df['status'].value_counts()
    n_overloaded_critical = counts.get('🟠 OVERLOADED', 0) + counts.get('🔴 CRITICAL', 0)
    print(f"[{title_prefix or network}] Snapshot: {snap} | "
          f"Overloaded/Critical: {n_overloaded_critical} | "
          f"Most loaded: {df.iloc[0]['line']} ({df.iloc[0]['loading_pct']:.1f}%)")

    # ── Plot ───────────────────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(16, max(6, len(df) * 0.28)))
        fig.patch.set_facecolor('#f5f5f0')

    color_map = {'🔴 CRITICAL': '#cc0000', '🟠 OVERLOADED': '#ff6600',
                 '🟡 WARNING': '#ffaa00', '🟢 OK': '#2ca02c'}
    colors = df['status'].map(color_map)

    bars = ax.barh(df['line'], df['loading_pct'], color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax.axvline(limit_pct, color='#ff6600', lw=1.8, linestyle='--', label=f'Security limit ({limit_pct}%)')
    ax.axvline(100, color='#cc0000', lw=1.8, label='Thermal limit (100%)')

    for bar, val in zip(bars, df['loading_pct']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=7, color='#333333')

    ax.set_xlabel('Line Loading (%)', fontsize=10)
    ax.set_title(f'{title_prefix}\nSnapshot: {snap} | Overloaded: {n_overloaded_critical}',
                 fontsize=11, fontweight='bold', color='#1a1a2e')
    ax.set_xlim(0, max(df['loading_pct'].max() * 1.15, 110))
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', linestyle='--', linewidth=0.4, color='#cccccc', alpha=0.8)
    ax.tick_params(labelsize=7)

    if standalone:
        plt.tight_layout()
        plt.show()

    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def line_utilization_share(network, thresholds=(75.0,), use_s_nom_opt=True,
                           plot=True, ax=None, title_prefix="", plot_threshold=None):
    """
    Percentage of the year each line is loaded above given % of its rated capacity.

    Parameters
    ----------
    network        : pypsa.Network (after pf() or optimize())
    thresholds     : float or iterable of float. Loading levels in % of rating,
                     e.g. (50, 75, 90, 100). A snapshot counts when
                     |p0| / rating * 100 >= threshold.
    use_s_nom_opt  : bool. True uses s_nom_opt (falls back to s_nom where it is
                     missing or zero); False uses s_nom.
    plot           : bool. Draw a horizontal bar chart for one threshold.
    ax             : matplotlib Axes or None.
    title_prefix   : str. Prepended to the plot title (e.g. scenario name).
    plot_threshold : float or None. Threshold shown in the plot
                     (None = the first threshold).

    Returns
    -------
    df : DataFrame indexed by line with
         - rating_MVA
         - mean_loading_pct, max_loading_pct
         - hours_above_<t>  and  pct_year_above_<t>  for each threshold t
         sorted by the first threshold's pct_year, descending.
         A final 'AVERAGE' row gives the mean across lines.
    """
    if network.lines_t.p0.empty:
        raise ValueError("network.lines_t.p0 is empty. Run network.pf() or network.optimize() first.")

    if np.isscalar(thresholds):
        thresholds = (thresholds,)
    thresholds = [float(t) for t in thresholds]

    # ── Rating ────────────────────────────────────────────────────────────────
    rating = network.lines['s_nom'].copy()
    if use_s_nom_opt and 's_nom_opt' in network.lines.columns:
        opt = network.lines['s_nom_opt']
        rating = opt.where(opt > 0, network.lines['s_nom'])
    rating = rating.replace(0, np.nan)

    # ── Loading per snapshot (% of rating) ────────────────────────────────────
    flow = network.lines_t.p0.abs()
    loading = flow.div(rating, axis=1) * 100

    # ── Snapshot weights (hours each snapshot represents) ─────────────────────
    w = network.snapshot_weightings
    weights = w['objective'] if 'objective' in w.columns else w.iloc[:, 0]
    weights = weights.reindex(loading.index).fillna(1.0)
    total_hours = weights.sum()

    # ── Build results ─────────────────────────────────────────────────────────
    df = pd.DataFrame({
        'rating_MVA': rating.round(1),
        'mean_loading_pct': loading.mul(weights, axis=0).sum().div(total_hours).round(2),
        'max_loading_pct': loading.max().round(2),
    })

    for t in thresholds:
        hours = (loading >= t).mul(weights, axis=0).sum()
        df[f'hours_above_{t:g}'] = hours.round(1)
        df[f'pct_year_above_{t:g}'] = (hours / total_hours * 100).round(2)

    first_col = f'pct_year_above_{thresholds[0]:g}'
    df = df.sort_values(first_col, ascending=False)

    avg = df.mean(numeric_only=True).round(2)
    avg.name = 'AVERAGE'
    df = pd.concat([df, avg.to_frame().T])

    # ── Console summary ───────────────────────────────────────────────────────
    label = title_prefix or "network"
    summary = " | ".join(f">={t:g}%: {df.loc['AVERAGE', f'pct_year_above_{t:g}']:.1f}% of year"
                         for t in thresholds)
    print(f"[{label}] Average across {len(df) - 1} lines, {total_hours:.0f} h | {summary}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        pt = float(plot_threshold) if plot_threshold is not None else thresholds[0]
        col = f'pct_year_above_{pt:g}'
        if col not in df.columns:
            raise ValueError(f"plot_threshold {pt:g} not in thresholds {thresholds}")

        data = df.drop(index='AVERAGE')[col].sort_values()
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.3)))

        colors = plt.cm.RdYlGn_r(data.values / 100.0)
        bars = ax.barh(data.index, data.values, color=colors, edgecolor='white', height=0.7)
        ax.axvline(df.loc['AVERAGE', col], color='#333333', lw=1.5, linestyle='--',
                   label=f"Average ({df.loc['AVERAGE', col]:.1f}%)")
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=7)

        ax.set_xlim(0, 105)
        ax.set_xlabel(f'% of year loaded at or above {pt:g}% of rated capacity')
        ax.set_title(f'{title_prefix}\nTime above {pt:g}% of rating', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.8)
        ax.tick_params(labelsize=7)
        if standalone:
            plt.tight_layout()
            plt.show()

    return df


