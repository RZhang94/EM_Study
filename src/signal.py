# functions to generate a signal pulse and calculate transmission time and phase

import numpy as np
import scipy.signal as sig
from typing import Dict, List, Tuple

SPEED_OF_LIGHT = 299792458 # m/s

class Position:
    # spatial position in space, meters
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, pos: 'Position') -> float:
        # calculate the distance between two positions
        return np.sqrt((self.x - pos.x)**2 + (self.y - pos.y)**2 + (self.z - pos.z)**2)

class Signal:
    # signal pulse at a point in space and time.
    # `phase` is this signal's own starting phase at its point of origin --
    # it resets to 0 on every reflection/transmission, since there is no
    # meaningful shared baseline to carry forward between two different
    # Signal objects.
    def __init__(self, name: str, frequency: float, time0: float, timef: float, position: Position, amplitude: float = 1, phase: float = 0):
        self.name = name
        self.frequency = frequency
        self.time0 = time0
        self.timef = timef
        self.pos = position
        self.phase = phase
        self.amp = amplitude

    def transmission(self, new_pos: Position, new_name: str, transmission_factor: float = 1, relative_permittivity: float = 1) -> 'Signal':
        # calculate the time of the signal transmission
        distance = self.pos.distance(new_pos)
        t = distance / (SPEED_OF_LIGHT / np.sqrt(relative_permittivity))
        # amplitude
        amp = self.amp * transmission_factor
        return Signal(new_name, self.frequency, t + self.time0, t+ self.timef, new_pos, amp, phase=0)
    
class PhaseShift:
    # provide the first Signal, the baseline signal, and then a list of other signals to compare against
    def __init__(self, baseline: Signal, other_signals: List[Signal], time: float):
        self.baseline = baseline
        self.other_signals = other_signals
        self.time = time
        self.phase_shifts = []
        self.amplitudes = []
        self.iq = []
        self.__calculate_phase_shifts()
        self.__calculate_iq()

    def __calculate_phase_shifts(self):
        # calculate the phase shifts for each other signal
        for other in self.other_signals:
            if self.time < other.time0 or self.time > other.timef:
                continue
            else:
                self.phase_shifts.append(self.__calculate_phase_shift__(other))
                self.amplitudes.append(other.amp)

    def __calculate_iq(self):
        # I_n = cos(phi_n), Q_n = -sin(phi_n), so that I_n + jQ_n ==
        # exp(-j*phi_n) (Euler's formula), matching the C_n = A_n*exp(-j*phi_n)
        # convention used in propagation/array literature
        for i in range(len(self.phase_shifts)):
            i_value = self.amplitudes[i] * np.cos(self.phase_shifts[i])
            q_value = -self.amplitudes[i] * np.sin(self.phase_shifts[i])
            self.iq.append([i_value, q_value])

    def __calculate_phase_shift__(self, other: Signal) -> float:
        # calculate the phase shift between this Sigal and other using tiem based
        delta_t = other.time0 - self.baseline.time0
        phase_shift = delta_t * 2 * np.pi * self.baseline.frequency
        phase_shift %= 2 * np.pi
        return phase_shift


def frequency_range(start_frequency: float, end_frequency: float, step: float) -> np.ndarray:
    # inclusive frequency sweep from start_frequency to end_frequency in
    # increments of step; uses linspace with a computed point count rather
    # than arange so end_frequency is always included despite float rounding
    steps = round((end_frequency - start_frequency) / step)
    return np.linspace(start_frequency, end_frequency, steps + 1)


def sweep_phase_shifts(
    frequencies,
    origin: Position,
    targets: List[Tuple[Position, str]],
    time: float,
    relative_permittivity: float = 1,
    tx_name: str = "tx",
    time0: float = 0,
    timef: float = 5e-8,
) -> List[PhaseShift]:
    # one PhaseShift per frequency: a fresh baseline Signal at `origin` is
    # transmitted to each (position, name) in `targets`, then compared
    # against itself at `time`
    phase_shifts = []
    for frequency in frequencies:
        baseline = Signal(tx_name, frequency, time0, timef, origin)
        others = [
            baseline.transmission(position, name, relative_permittivity=relative_permittivity)
            for position, name in targets
        ]
        phase_shifts.append(PhaseShift(baseline, others, time))
    return phase_shifts


def combine_phase_shifts_by_name(phase_shifts: List[PhaseShift]) -> Dict[str, List[dict]]:
    # group each PhaseShift's per-`other` data (phase shift, amplitude, IQ)
    # by the other signal's name across a frequency sweep, sorted by
    # frequency -- replicates PhaseShift's own time-window filter so each
    # entry in phase_shifts/amplitudes/iq lines up with the `other` signal
    # it was computed from
    series = {}
    for phase_shift in phase_shifts:
        frequency = phase_shift.baseline.frequency
        included = [
            other for other in phase_shift.other_signals
            if phase_shift.time >= other.time0 and phase_shift.time <= other.timef
        ]
        for other, shift, amplitude, iq in zip(
            included, phase_shift.phase_shifts, phase_shift.amplitudes, phase_shift.iq
        ):
            series.setdefault(other.name, []).append({
                "frequency": frequency,
                "phase_shift": shift,
                "amplitude": amplitude,
                "i": iq[0],
                "q": iq[1],
            })

    for points in series.values():
        points.sort(key=lambda point: point["frequency"])

    return series


def resultant_iq(phase_shift: PhaseShift) -> dict:
    # combine every `other` signal in `phase_shift` (e.g. multiple arrival
    # paths at one frequency) into a single resultant two ways:
    #   - exponential_sum: sum of each path's amplitude * e^(-j * phase)
    #   - vector_sum: sum of each path's already-computed I/Q components
    #     (I = amplitude * cos(phase), Q = -amplitude * sin(phase), i.e.
    #     amplitude * e^(-j * phase)) added as (I + jQ)
    # both now use the same e^(-j*phase) convention, so they are identical
    # (exact, not just in magnitude) -- kept as two methods for verification
    amplitudes = np.asarray(phase_shift.amplitudes, dtype=float)
    phases = np.asarray(phase_shift.phase_shifts, dtype=float)
    exponential_sum = np.sum(amplitudes * np.exp(-1j * phases))

    iq = np.asarray(phase_shift.iq, dtype=float).reshape(-1, 2)
    vector_sum = np.sum(iq[:, 0]) + 1j * np.sum(iq[:, 1])

    return {
        "frequency": phase_shift.baseline.frequency,
        "exponential_sum": exponential_sum,
        "vector_sum": vector_sum,
    }


def resultant_iq_sweep(phase_shifts: List[PhaseShift]) -> List[dict]:
    # resultant_iq() for every PhaseShift in a frequency sweep, sorted by
    # frequency
    return sorted(
        (resultant_iq(phase_shift) for phase_shift in phase_shifts),
        key=lambda result: result["frequency"],
    )


def hann_window(n_points: int) -> np.ndarray:
    # w[n] = 0.5 - 0.5*cos(2*pi*n / (n_points - 1)), the standard
    # Hann/raised-cosine taper. For this project's canonical 0.5-3 GHz,
    # 25 MHz-step sweep (101 points, n_points - 1 == 100), this is exactly
    # w[n] = 0.5 - 0.5*cos(2*pi*n / 100). Equal to tukey_window(n_points, 1.0).
    n = np.arange(n_points)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (n_points - 1))


def tukey_window(n_points: int, alpha: float = 0.5) -> np.ndarray:
    # tapered-cosine window: flat (untapered, value 1) across the middle
    # (1 - alpha) fraction, cosine-tapered over the outer alpha fraction
    # split between both edges. alpha=0 -> rectangular (no taper, noisiest,
    # narrowest mainlobe); alpha=1 -> exactly hann_window (fully tapered).
    return sig.windows.tukey(n_points, alpha=alpha)


def range_profile(
    resultant_iq_results: List[dict],
    relative_permittivity: float = 1,
    target_points: int = 1024,
    apply_window: bool = True,
    window_alpha: float = 0.5,
) -> dict:
    # IFFT the combined-echo spectrum (resultant_iq_sweep's exponential_sum,
    # C_n = A_n*exp(-j*phi_n)) across a uniformly-stepped frequency sweep
    # into a delay/range profile -- the stepped-frequency-radar technique for
    # resolving individual echoes that overlap in the frequency domain.
    # Requires a uniform frequency step (e.g. from frequency_range()); the
    # IFFT's synthesis kernel e^(+j*...) is the correct inverse of the
    # e^(-j*phi) delay convention used above, so peaks land at the true
    # positive delay/range of each echo rather than mirrored/aliased.
    #
    # if apply_window, a Tukey taper (zw[n] = w[n]*z[n]) is applied to the
    # real, unpadded spectrum before zero-padding, trading some sidelobe
    # suppression for a narrower main peak than the full Hann window --
    # `window_alpha` dials how much taper is applied (0 = none/noisiest,
    # 1 = full Hann) -- the standard apodization step before an SFCW IFFT.
    #
    # zeros are appended to the (tapered) spectrum before the IFFT until it
    # reaches `target_points` long (a no-op if the sweep already has that
    # many points). This is standard zero-padding, not extra measurements:
    # it sinc-interpolates a finer-sampled, smoother-looking profile for
    # display, but it does not improve true range resolution -- that's
    # still set by the real sweep bandwidth (1 / (N * step)).
    frequencies = np.array([result["frequency"] for result in resultant_iq_results])
    spectrum = np.array([result["exponential_sum"] for result in resultant_iq_results])

    if apply_window:
        spectrum = spectrum * tukey_window(len(spectrum), alpha=window_alpha)

    step = frequencies[1] - frequencies[0]
    pad_points = max(0, target_points - len(spectrum))
    padded_spectrum = np.pad(spectrum, (0, pad_points))

    delay = np.fft.fftshift(np.fft.fftfreq(len(padded_spectrum), d=step))
    impulse_response = np.fft.fftshift(np.fft.ifft(padded_spectrum))

    velocity = SPEED_OF_LIGHT / np.sqrt(relative_permittivity)
    return {
        "delay": delay,
        "range": delay * velocity,
        "impulse_response": impulse_response,
    }


