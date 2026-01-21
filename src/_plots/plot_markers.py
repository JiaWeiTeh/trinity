#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot markers helper module for TRINITY paper plots.

Provides consistent vertical line markers for:
- Phase transitions (energy -> transition, transition -> momentum)
- R2 > rCloud breakout
- Collapse onset (isCollapse becomes True)

Supports multiple datasets on the same subplot with color differentiation.

Usage:
    from plot_markers import add_plot_markers, get_marker_legend_handles

    # Single dataset
    add_plot_markers(ax, t, phase, R2, rcloud, isCollapse)

    # Multiple datasets (e.g., comparing two runs)
    add_plot_markers(ax, t1, phase1, R2_1, rcloud1, isCollapse1,
                     dataset_label="Original", dataset_color="blue")
    add_plot_markers(ax, t2, phase2, R2_2, rcloud2, isCollapse2,
                     dataset_label="Modified", dataset_color="red")

    # Add legend handles
    handles = get_marker_legend_handles(include_collapse=True)

Author: TRINITY Team
"""

import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D


# Default marker styles
MARKER_STYLES = {
    'transition': {
        'color': 'red',
        'lw': 2,
        'alpha': 0.2,
        'label': 'T',
        'label_y': 0.95,
    },
    'momentum': {
        'color': 'red',
        'lw': 2,
        'alpha': 0.2,
        'label': 'M',
        'label_y': 0.95,
    },
    'rcloud': {
        'color': 'black',
        'ls': '--',
        'lw': 1.6,
        'alpha': 0.25,
        'label': r'$R_2 = R_{\rm cloud}$',
        'label_y': 0.05,
    },
    'collapse': {
        'color': 'purple',
        'ls': '--',
        'lw': 1.8,
        'alpha': 0.6,
        'label': 'Collapse',
        'label_y': 0.05,
    },
}


def find_phase_transitions(t, phase):
    """
    Find times when phase transitions occur.

    Parameters
    ----------
    t : array-like
        Time array
    phase : array-like
        Phase array (strings like 'energy', 'implicit', 'transition', 'momentum')

    Returns
    -------
    dict
        Dictionary with keys:
        - 't_transition': list of times entering transition phase
        - 't_momentum': list of times entering momentum phase
    """
    t = np.asarray(t)
    phase = np.asarray(phase)

    result = {
        't_transition': [],
        't_momentum': [],
    }

    if len(t) < 2:
        return result

    # Find energy/implicit -> transition
    idx_T = np.flatnonzero(
        np.isin(phase[:-1], ["energy", "implicit"]) & (phase[1:] == "transition")
    ) + 1
    result['t_transition'] = list(t[idx_T])

    # Find transition -> momentum
    idx_M = np.flatnonzero(
        (phase[:-1] == "transition") & (phase[1:] == "momentum")
    ) + 1
    result['t_momentum'] = list(t[idx_M])

    return result


def find_rcloud_crossing(t, R2, rcloud):
    """
    Find time when R2 first exceeds rCloud.

    Parameters
    ----------
    t : array-like
        Time array
    R2 : array-like
        Outer shell radius array
    rcloud : float
        Cloud radius

    Returns
    -------
    float or None
        Time of first crossing, or None if never crossed
    """
    if not np.isfinite(rcloud):
        return None

    t = np.asarray(t)
    R2 = np.asarray(R2)

    idx = np.flatnonzero(np.isfinite(R2) & (R2 > rcloud))
    if idx.size:
        return t[idx[0]]
    return None


def find_collapse_time(t, isCollapse):
    """
    Find time when collapse first begins.

    Parameters
    ----------
    t : array-like
        Time array
    isCollapse : array-like
        Boolean or boolean-like array indicating collapse state

    Returns
    -------
    float or None
        Time of collapse onset, or None if no collapse
    """
    if isCollapse is None:
        return None

    t = np.asarray(t)
    collapse_mask = np.array([bool(c) for c in isCollapse])

    idx = np.flatnonzero(collapse_mask)
    if idx.size:
        return t[idx[0]]
    return None


def add_phase_markers(ax, t, phase, dataset_label=None, dataset_color=None,
                      label_pad_points=4, show_labels=True):
    """
    Add phase transition markers (T for transition, M for momentum).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    t : array-like
        Time array
    phase : array-like
        Phase array
    dataset_label : str, optional
        Label for this dataset (used when multiple datasets on same plot)
    dataset_color : str, optional
        Color to use for markers (overrides default if set)
    label_pad_points : int
        Padding in points for text labels
    show_labels : bool
        Whether to show "T" and "M" text labels

    Returns
    -------
    dict
        Dictionary with transition times found
    """
    fig = ax.figure
    transitions = find_phase_transitions(t, phase)

    # Determine colors
    t_style = MARKER_STYLES['transition'].copy()
    m_style = MARKER_STYLES['momentum'].copy()

    if dataset_color is not None:
        t_style['color'] = dataset_color
        m_style['color'] = dataset_color
        t_style['alpha'] = 0.4  # Increase alpha when using custom colors
        m_style['alpha'] = 0.4

    # Add transition phase markers
    for x in transitions['t_transition']:
        ax.axvline(x, color=t_style['color'], lw=t_style['lw'],
                   alpha=t_style['alpha'], zorder=0)
        if show_labels:
            label_text = "T"
            if dataset_label:
                label_text = f"T ({dataset_label})"
            ax.text(
                x, t_style['label_y'], label_text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color=t_style['color'], alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

    # Add momentum phase markers
    for x in transitions['t_momentum']:
        ax.axvline(x, color=m_style['color'], lw=m_style['lw'],
                   alpha=m_style['alpha'], zorder=0)
        if show_labels:
            label_text = "M"
            if dataset_label:
                label_text = f"M ({dataset_label})"
            ax.text(
                x, m_style['label_y'], label_text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color=m_style['color'], alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

    return transitions


def add_rcloud_marker(ax, t, R2, rcloud, dataset_label=None, dataset_color=None,
                      label_pad_points=4, show_label=True):
    """
    Add R2 > rCloud breakout marker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add marker to
    t : array-like
        Time array
    R2 : array-like
        Outer shell radius array
    rcloud : float
        Cloud radius
    dataset_label : str, optional
        Label for this dataset
    dataset_color : str, optional
        Color to use for marker
    label_pad_points : int
        Padding in points for text label
    show_label : bool
        Whether to show text label

    Returns
    -------
    float or None
        Time of crossing, or None if not found
    """
    fig = ax.figure
    t_cross = find_rcloud_crossing(t, R2, rcloud)

    if t_cross is None:
        return None

    style = MARKER_STYLES['rcloud'].copy()
    if dataset_color is not None:
        style['color'] = dataset_color

    ax.axvline(t_cross, color=style['color'], ls=style['ls'],
               lw=style['lw'], alpha=style['alpha'], zorder=0)

    if show_label:
        text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
            label_pad_points/72, 0, fig.dpi_scale_trans
        )
        label_text = style['label']
        if dataset_label:
            label_text = f"{style['label']} ({dataset_label})"
        ax.text(
            t_cross, style['label_y'], label_text,
            transform=text_trans,
            ha="left", va="bottom",
            fontsize=8, color=style['color'], alpha=0.8,
            rotation=90, zorder=5
        )

    return t_cross


def add_collapse_marker(ax, t, isCollapse, dataset_label=None, dataset_color=None,
                        label_pad_points=4, show_label=True):
    """
    Add collapse onset marker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add marker to
    t : array-like
        Time array
    isCollapse : array-like
        Boolean array indicating collapse state
    dataset_label : str, optional
        Label for this dataset
    dataset_color : str, optional
        Color to use for marker (default: purple)
    label_pad_points : int
        Padding in points for text label
    show_label : bool
        Whether to show text label

    Returns
    -------
    float or None
        Time of collapse, or None if not found
    """
    fig = ax.figure
    t_collapse = find_collapse_time(t, isCollapse)

    if t_collapse is None:
        return None

    style = MARKER_STYLES['collapse'].copy()
    if dataset_color is not None:
        style['color'] = dataset_color

    ax.axvline(t_collapse, color=style['color'], ls=style['ls'],
               lw=style['lw'], alpha=style['alpha'], zorder=0)

    if show_label:
        text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
            label_pad_points/72, 0, fig.dpi_scale_trans
        )
        label_text = style['label']
        if dataset_label:
            label_text = f"{style['label']} ({dataset_label})"
        ax.text(
            t_collapse, style['label_y'], label_text,
            transform=text_trans,
            ha="left", va="bottom",
            fontsize=8, color=style['color'], alpha=0.8,
            rotation=90, zorder=5
        )

    return t_collapse


def add_plot_markers(ax, t, phase=None, R2=None, rcloud=None, isCollapse=None,
                     dataset_label=None, dataset_color=None,
                     show_phase=True, show_rcloud=True, show_collapse=True,
                     show_labels=True, label_pad_points=4):
    """
    Add all plot markers (phase transitions, rcloud crossing, collapse).

    This is the main function to use for adding markers to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    t : array-like
        Time array
    phase : array-like, optional
        Phase array (for transition markers)
    R2 : array-like, optional
        Outer shell radius (for rcloud marker)
    rcloud : float, optional
        Cloud radius (for rcloud marker)
    isCollapse : array-like, optional
        Collapse state array (for collapse marker)
    dataset_label : str, optional
        Label for this dataset (for multi-dataset plots)
    dataset_color : str, optional
        Color for markers (for multi-dataset plots)
    show_phase : bool
        Whether to show phase transition markers
    show_rcloud : bool
        Whether to show R2 > rCloud marker
    show_collapse : bool
        Whether to show collapse marker
    show_labels : bool
        Whether to show text labels on markers
    label_pad_points : int
        Padding in points for text labels

    Returns
    -------
    dict
        Dictionary with all detected event times:
        - 't_transition': list of transition times
        - 't_momentum': list of momentum times
        - 't_rcloud': rcloud crossing time (or None)
        - 't_collapse': collapse time (or None)

    Examples
    --------
    # Single dataset (typical paper plot)
    add_plot_markers(ax, t, phase, R2, rcloud, isCollapse)

    # Multiple datasets with color differentiation
    add_plot_markers(ax, t1, phase1, R2_1, rcloud1, isCollapse1,
                     dataset_label="Run A", dataset_color="blue")
    add_plot_markers(ax, t2, phase2, R2_2, rcloud2, isCollapse2,
                     dataset_label="Run B", dataset_color="red")
    """
    result = {
        't_transition': [],
        't_momentum': [],
        't_rcloud': None,
        't_collapse': None,
    }

    # Phase transitions
    if show_phase and phase is not None:
        transitions = add_phase_markers(
            ax, t, phase,
            dataset_label=dataset_label,
            dataset_color=dataset_color,
            label_pad_points=label_pad_points,
            show_labels=show_labels
        )
        result['t_transition'] = transitions['t_transition']
        result['t_momentum'] = transitions['t_momentum']

    # R2 > rCloud
    if show_rcloud and R2 is not None and rcloud is not None:
        result['t_rcloud'] = add_rcloud_marker(
            ax, t, R2, rcloud,
            dataset_label=dataset_label,
            dataset_color=dataset_color,
            label_pad_points=label_pad_points,
            show_label=show_labels
        )

    # Collapse
    if show_collapse and isCollapse is not None:
        result['t_collapse'] = add_collapse_marker(
            ax, t, isCollapse,
            dataset_label=dataset_label,
            dataset_color=dataset_color,
            label_pad_points=label_pad_points,
            show_label=show_labels
        )

    return result


def get_marker_legend_handles(include_phase=True, include_rcloud=True,
                               include_collapse=True):
    """
    Get legend handles for plot markers.

    Parameters
    ----------
    include_phase : bool
        Include phase change marker in legend
    include_rcloud : bool
        Include R2 > rCloud marker in legend
    include_collapse : bool
        Include collapse marker in legend

    Returns
    -------
    list
        List of Line2D handles for legend
    """
    handles = []

    if include_phase:
        style = MARKER_STYLES['transition']
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            lw=style['lw'],
            alpha=0.3,
            label="Phase change (T/M)"
        ))

    if include_rcloud:
        style = MARKER_STYLES['rcloud']
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            ls=style['ls'],
            lw=style['lw'],
            alpha=style['alpha'],
            label=r"$R_2 > R_{\rm cloud}$"
        ))

    if include_collapse:
        style = MARKER_STYLES['collapse']
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            ls=style['ls'],
            lw=style['lw'],
            alpha=style['alpha'],
            label="Collapse"
        ))

    return handles


def get_multi_dataset_legend_handles(datasets, include_phase=True,
                                      include_rcloud=True, include_collapse=True):
    """
    Get legend handles for multi-dataset plots with color differentiation.

    Parameters
    ----------
    datasets : list of dict
        List of dataset info dicts with keys 'label' and 'color'
        Example: [{'label': 'Original', 'color': 'blue'},
                  {'label': 'Modified', 'color': 'red'}]
    include_phase : bool
        Include phase markers in legend
    include_rcloud : bool
        Include rcloud markers in legend
    include_collapse : bool
        Include collapse markers in legend

    Returns
    -------
    list
        List of Line2D handles for legend
    """
    handles = []

    for ds in datasets:
        label = ds.get('label', 'Dataset')
        color = ds.get('color', 'black')

        if include_phase:
            handles.append(Line2D(
                [0], [0],
                color=color,
                lw=2,
                alpha=0.4,
                label=f"Phase change ({label})"
            ))

    # Add generic markers that apply to all datasets
    if include_rcloud:
        handles.append(Line2D(
            [0], [0],
            color='black',
            ls='--',
            lw=1.6,
            alpha=0.25,
            label=r"$R_2 > R_{\rm cloud}$"
        ))

    if include_collapse:
        handles.append(Line2D(
            [0], [0],
            color='purple',
            ls='--',
            lw=1.8,
            alpha=0.6,
            label="Collapse"
        ))

    return handles
