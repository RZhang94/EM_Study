# plot visualizations.
import numpy as np
import matplotlib.pyplot as plt

from src.signal import (
    Position,
    Signal,
    PhaseShift,
    frequency_range,
    sweep_phase_shifts,
    combine_phase_shifts_by_name,
    resultant_iq_sweep,
    range_profile,
)


def plot_signals(signals, points_per_period=25, max_points=20_000, ax=None, show=True):
    if ax is None:
        _, ax = plt.subplots()

    for s in signals:
        duration = s.timef - s.time0
        wanted_points = duration * s.frequency * points_per_period

        if wanted_points > max_points:
            # the full duration has too many cycles to sample at
            # points_per_period without exceeding max_points -- show as many
            # full, correctly-resolved cycles as will fit instead of
            # aliasing the whole duration into too few samples.
            periods_shown = max_points / points_per_period
            plot_duration = periods_shown / s.frequency
            print(
                f"{s.name}: {duration * s.frequency:.0f} cycles over {duration:.3g}s "
                f"can't be resolved in {max_points} points; showing the first "
                f"{periods_shown:.0f} cycles only"
            )
        else:
            plot_duration = duration

        num_points = max(2, min(int(wanted_points), max_points))
        t = np.linspace(0, s.timef - s.time0 + plot_duration, num_points)
        waveform = s.amp * np.sin(2 * np.pi * s.frequency * t + s.phase)
        t += s.time0
        ax.plot(t, waveform, label=s.name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()

    if show:
        plt.show()

    return ax


def plot_signal_groups(groups, points_per_period=50, max_points=20_000, show=True):
    # one row of subplots per group, each group plotted as its own
    # plot_signals() call so every row can hold however many signals it needs
    fig, axes = plt.subplots(nrows=len(groups), ncols=1, sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, group in zip(axes, groups):
        plot_signals(group, points_per_period=points_per_period, max_points=max_points, ax=ax, show=False)

    fig.tight_layout()

    if len(axes) > 1:
        # pull the rows close together; leave the last one alone so it keeps
        # its own x-axis label and tick labels. Must run after tight_layout,
        # which would otherwise reset hspace back out.
        for ax in axes[:-1]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        fig.subplots_adjust(hspace=0.1)

    if show:
        plt.show()

    return axes


_SERIES_COLORS = ["#4C6EF5", "#F76707", "#2F9E44", "#AE3EC9"]


def plot_phase_lag_vs_frequency(series, relative_permittivity, ax=None, show=True):
    # `series` is the output of signal.combine_phase_shifts_by_name: one
    # series (name -> sorted list of per-frequency dicts) per `other` signal
    # name (e.g. "rx_a" vs "rx_b")
    if ax is None:
        _, ax = plt.subplots()

    for (name, points), color in zip(series.items(), _SERIES_COLORS):
        frequencies = [point["frequency"] for point in points]
        phase_lags_deg = [np.degrees(point["phase_shift"]) for point in points]
        ax.plot(frequencies, phase_lags_deg, marker="o", color=color, label=name)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase Lag (deg)")
    ax.set_title("Phase Lag vs Frequency")
    ax.legend()

    # mark the relative permittivity used for this sweep on the first series
    first_points = next(iter(series.values()))
    mid = len(first_points) // 2
    ax.annotate(
        f"εᵣ = {relative_permittivity:g}",
        xy=(first_points[mid]["frequency"], np.degrees(first_points[mid]["phase_shift"])),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    if show:
        plt.show()

    return ax


def plot_iq_vs_frequency(series, show=True):
    # `series` is the output of signal.combine_phase_shifts_by_name; I and Q
    # get their own stacked subplot (never a dual-axis single plot) so both
    # stay readable on their own scale
    fig, (ax_i, ax_q) = plt.subplots(nrows=2, ncols=1, sharex=True)

    for (name, points), color in zip(series.items(), _SERIES_COLORS):
        frequencies = [point["frequency"] for point in points]
        i_values = [point["i"] for point in points]
        q_values = [point["q"] for point in points]
        ax_i.plot(frequencies, i_values, marker="o", color=color, label=name)
        ax_q.plot(frequencies, q_values, marker="o", color=color, label=name)

    ax_i.set_ylabel("I")
    ax_i.set_title("IQ vs Frequency")
    ax_i.legend()

    ax_q.set_xlabel("Frequency (Hz)")
    ax_q.set_ylabel("Q")

    fig.tight_layout()

    if show:
        plt.show()

    return ax_i, ax_q


def plot_combined_iq_vs_frequency(resultant, show=True):
    # `resultant` is the output of signal.resultant_iq_sweep: every path at
    # each frequency already summed into one complex value (C_n), so this is
    # a single I and single Q series rather than one per path name
    frequencies = [result["frequency"] for result in resultant]
    i_values = [result["exponential_sum"].real for result in resultant]
    q_values = [result["exponential_sum"].imag for result in resultant]

    fig, (ax_i, ax_q) = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax_i.plot(frequencies, i_values, marker="o", color=_SERIES_COLORS[0])
    ax_q.plot(frequencies, q_values, marker="o", color=_SERIES_COLORS[0])

    ax_i.set_ylabel("I")
    ax_i.set_title("Combined IQ vs Frequency")

    ax_q.set_xlabel("Frequency (Hz)")
    ax_q.set_ylabel("Q")

    fig.tight_layout()

    if show:
        plt.show()

    return ax_i, ax_q


def plot_range_profile(profile, ax=None, show=True, label=None, color="#4C6EF5"):
    # `profile` is the output of signal.range_profile: the combined (summed)
    # echoes from the frequency sweep, deconvolved by IFFT into a
    # delay/range-domain profile -- i.e. what a stepped-frequency radar would
    # display as its reflected echoes
    if ax is None:
        _, ax = plt.subplots()

    # normalized for display so peak echo = 1, independent of how many
    # zero-padding points range_profile() used (ifft's 1/N scaling would
    # otherwise make the raw magnitude incomparable across pad amounts)
    magnitude = np.abs(profile["impulse_response"])
    magnitude = magnitude / np.max(magnitude)

    # only positive range is physical (a reflector in front of the radar);
    # the negative half is fftshift's mirror of the same real-valued delay
    # axis, not a separate measurement. Capped at 0.5m to match this
    # project's target depths.
    in_range = (profile["range"] >= 0) & (profile["range"] <= 0.5)
    range_values = profile["range"][in_range]
    magnitude = magnitude[in_range]

    ax.plot(range_values, magnitude, color=color, label=label)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalized |Reflected Echo|")
    ax.set_title("Deconvolved Echo Profile")

    if label is not None:
        ax.legend()

    if show:
        plt.show()

    return ax


def plot_range_profile_comparison(
    resultant,
    alphas=(0.25, 0.5, 1.0),
    relative_permittivity=1,
    target_points=1024,
    ax=None,
    show=True,
):
    # overlays range_profile() at several Tukey window_alpha values on one
    # axes, so the sidelobe-suppression/mainlobe-width tradeoff is directly
    # comparable
    if ax is None:
        _, ax = plt.subplots()

    for alpha, color in zip(alphas, _SERIES_COLORS):
        profile = range_profile(
            resultant,
            relative_permittivity=relative_permittivity,
            target_points=target_points,
            apply_window=True,
            window_alpha=alpha,
        )
        plot_range_profile(profile, ax=ax, show=False, label=f"α={alpha:g}", color=color)

    ax.set_title("Deconvolved Echo Profile vs. Tukey α")

    if show:
        plt.show()

    return ax


if __name__ == "__main__":
    tx = Signal("tx", frequency=3e9, time0=0, timef=5e-8, position=Position(0, 0, 0))
    a = tx.transmission(Position(0, 0, 0.08), "a", relative_permittivity=9)
    b = tx.transmission(Position(0, 0, 0.05), "b", relative_permittivity=9)
    rx_a = a.transmission(tx.pos, "rx_a", relative_permittivity=9)
    rx_b = b.transmission(tx.pos, "rx_b", relative_permittivity=9)
    # plot_signal_groups([[tx], [a, b], [rx_a, rx_b]])

    phase_shift = PhaseShift(tx, [rx_a, rx_b], 2e-8)

    sweep_relative_permittivity = 9
    phase_shifts = sweep_phase_shifts(
        frequency_range(0.5e9, 3e9, 25e6),
        origin=Position(0, 0, 0),
        targets=[(Position(0, 0, 0.14), "rx"), (Position(0, 0, 0.10), "rx_b")],
        time=2e-8,
        relative_permittivity=sweep_relative_permittivity,
    )
    series = combine_phase_shifts_by_name(phase_shifts)

    plot_phase_lag_vs_frequency(series, relative_permittivity=sweep_relative_permittivity)
    plot_iq_vs_frequency(series)

    resultant = resultant_iq_sweep(phase_shifts)
    plot_combined_iq_vs_frequency(resultant)

    profile = range_profile(resultant, relative_permittivity=sweep_relative_permittivity)
    plot_range_profile(profile)

    plot_range_profile_comparison(
        resultant,
        alphas=(0.25, 0.5, 1.0),
        relative_permittivity=sweep_relative_permittivity,
    )

    print("done")
