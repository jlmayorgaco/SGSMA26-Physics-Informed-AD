"""Plotting helpers for m3 calibration outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src.calibration.calibration_metrics import sample_rate_from_t


def make_calibration_plot(bus_id, signal_key, chunk_id, scope, t, real, prepared, metrics, support_status, profile_status, out_path):
    n = min(500, len(t), len(real))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Bus {bus_id} | {signal_key} | {scope}:{chunk_id} | KS={metrics['ks_stat']:.4f} "
        f"p={metrics.get('ks_pvalue', 0.0):.2e}\nSupport={support_status} | Profile={profile_status}",
        fontsize=9,
    )
    axes[0].plot(t[:n], prepared["sim_clean"][:n], lw=1.0, label="ANDES clean fit")
    axes[0].plot(t[:n], prepared["sim_drifted"][:n], lw=0.9, alpha=0.85, label="clean + operating drift")
    axes[0].plot(t[:n], prepared["sim_noisy"][:n], lw=0.8, alpha=0.75, label="drift + raw noise")
    if n > 0:
        axes[0].plot(np.linspace(t[0], t[min(n - 1, len(t) - 1)], n), real[:n], lw=0.8, alpha=0.75, label="raw event0")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].hist(real, bins=50, density=True, alpha=0.55, label="raw event0")
    axes[1].hist(prepared["sim_noisy"], bins=50, density=True, alpha=0.55, label="ANDES+drift+noise")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_spectral_plot(bus_id, signal_key, t, real, prepared, out_path):
    fs = sample_rate_from_t(t)
    real = np.asarray(real, dtype=float)
    sim = np.asarray(prepared["sim_noisy"], dtype=float)
    real = real[np.isfinite(real)]
    sim = sim[np.isfinite(sim)]
    if len(real) < 8 or len(sim) < 8:
        return
    npr = min(512, len(real))
    nps = min(512, len(sim))
    fr, pr = welch(real - np.mean(real), fs=fs, nperseg=npr)
    fsim, ps = welch(sim - np.mean(sim), fs=fs, nperseg=nps)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(fr, pr + 1e-18, label="raw event0")
    ax.semilogy(fsim, ps + 1e-18, label="ANDES+drift+noise")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Bus {bus_id} | {signal_key} spectral match")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
