"""EMG preprocessing pipeline adapted from Clark et al. (2010).

Pipeline (applied per channel):
    1. Band-pass 40-450 Hz, 4th-order Butterworth, filtfilt
    2. Notch 60 Hz (Q=30)  -- KAIST power grid for Gait120
    3. Full-wave rectification (abs)
    4. Low-pass 4 Hz envelope, 4th-order Butterworth, filtfilt

Filtfilt applies the filter forward and backward, so the effective order
is 8 and the output has zero phase distortion. Clark 2010 used a
4th-order zero-lag Butterworth at the envelope stage; we match that.

**Clark 2010 p.845 (right column) reference spec**:
    "Muscle activation signals (EMGs) were high-pass filtered (40 Hz)
    with a zero-lag fourth-order Butterworth filter, demeaned, rectified,
    and smoothed with a zero-lag fourth-order low-pass filter (4 Hz)
    (Bradstreet 2007; Buchanan et al. 2004)."

**Deviations from Clark 2010 in this implementation**:
    (1) Band-pass 40-450 Hz (this file) vs high-pass 40 Hz only (Clark).
        Clark does not specify an upper cutoff; we add 450 Hz to further
        suppress motion artifact in the STS task, which has high trunk
        acceleration. Because the 4 Hz envelope low-pass (step 4)
        dominates the final smoothed output, the added upper cutoff is
        functionally near-equivalent in the envelope domain.
    (2) Notch 60 Hz (this file) vs not specified by Clark. Added for the
        Gait120 dataset, recorded in South Korea on a 60 Hz power grid.
    (3) Explicit demean step (Clark p.845) is omitted here. The preceding
        zero-lag band-pass at 40 Hz already removes the DC component, so
        the explicit demean is usually a no-op; add it for strict Clark
        alignment if downstream analysis depends on an exact zero-mean
        intermediate signal.

An earlier revision of this docstring incorrectly attributed "20-500 Hz
bandpass" to Clark 2010. Verified against Clark 2010 PDF p.845 right
column on 2026-04-16: Clark specifies high-pass 40 Hz only, no upper
cutoff, no notch. The previous attribution was wrong and has been
corrected.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def _butter_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Bandpass SOS (second-order sections) — numerically stable for order >= 4."""
    return butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")


def _butter_lowpass_sos(cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Lowpass SOS — preferred over ``ba`` form, especially for low cutoffs."""
    return butter(order, cutoff, btype="lowpass", fs=fs, output="sos")


def preprocess_emg(
    raw_emg: np.ndarray,
    fs: float = 2000.0,
    notch_freq: float = 60.0,
    bp_low: float = 40.0,
    bp_high: float = 450.0,
    env_cutoff: float = 4.0,
    pad_seconds: float = 1.0,
) -> np.ndarray:
    """EMG preprocessing adapted from Clark et al. (2010) p.845.

    Parameters
    ----------
    raw_emg : ndarray, shape (n_samples, n_channels)
        Raw EMG in microvolts or arbitrary units; all channels processed.
    fs : float
        Sampling rate (Hz). Gait120 uses 2000.
    notch_freq : float
        Power-line frequency. Korea / KAIST = 60 Hz. Note: Clark 2010
        does not specify a notch filter; this is a Gait120-specific
        addition.
    bp_low, bp_high : float
        Band-pass edges. Default 40-450 Hz. **Note: Clark 2010 p.845
        specifies high-pass 40 Hz only, with no upper cutoff.** The
        450 Hz upper edge is an added deviation to suppress motion
        artifact in the STS task (see module docstring for full
        rationale).
    env_cutoff : float
        Envelope low-pass cutoff. Clark 2010 p.845 specifies 4 Hz.
    pad_seconds : float
        Symmetric reflection padding on each edge to supplement
        sosfiltfilt's internal padding for extra edge-artifact
        suppression on the very-low-cutoff envelope stage; trimmed
        before returning.

    Returns
    -------
    envelope : ndarray, same shape as raw_emg, non-negative.
    """
    if raw_emg.ndim == 1:
        raw_emg = raw_emg[:, None]
    x = np.asarray(raw_emg, dtype=np.float64)

    # Reflection-pad to supplement sosfiltfilt's internal padding.
    # The 4 Hz envelope lowpass has a very long impulse response, so
    # extra padding beyond the default is beneficial.
    n_pad = int(round(pad_seconds * fs))
    if n_pad > 0 and x.shape[0] > 2 * n_pad:
        head = x[1 : n_pad + 1][::-1]
        tail = x[-n_pad - 1 : -1][::-1]
        x_pad = np.concatenate([head, x, tail], axis=0)
    else:
        n_pad = 0
        x_pad = x

    # 1. Band-pass 40-450 Hz (SOS form for order-4 stability)
    sos_bp = _butter_bandpass_sos(bp_low, bp_high, fs)
    bp = sosfiltfilt(sos_bp, x_pad, axis=0)

    # 2. Notch 60 Hz (KAIST). Converted to SOS for consistency with
    #    the rest of the pipeline (iirnotch is 2nd-order, so ba and SOS
    #    are numerically equivalent, but using SOS uniformly prevents
    #    accidental ba-form usage if harmonics are added later).
    b_n, a_n = iirnotch(notch_freq, Q=30, fs=fs)
    sos_notch = tf2sos(b_n, a_n)
    nf = sosfiltfilt(sos_notch, bp, axis=0)

    # 3. Full-wave rectification
    rect = np.abs(nf)

    # 4. Low-pass 4 Hz envelope (SOS form; low normalized cutoff is numerically delicate)
    sos_lp = _butter_lowpass_sos(env_cutoff, fs)
    envelope = sosfiltfilt(sos_lp, rect, axis=0)

    if n_pad > 0:
        envelope = envelope[n_pad:-n_pad]

    # NMF requires non-negative inputs; clip tiny numerical negatives.
    envelope = np.clip(envelope, 0.0, None)
    return envelope


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square error between two equal-shape arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))
