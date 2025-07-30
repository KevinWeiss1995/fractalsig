"""
Plotting utilities for FractalSig library.
Provides convenient visualization functions for fractional processes and their analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import warnings

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_fgn(data: np.ndarray, H: Optional[float] = None, title: Optional[str] = None,
             figsize: Tuple[int, int] = (12, 4), show_stats: bool = True,
             color: str = 'steelblue', alpha: float = 0.8) -> plt.Figure:
    """
    Plot fractional Gaussian noise time series.
    
    Parameters:
        data: fGn time series
        H: Hurst exponent for title (optional)
        title: Custom title (optional)
        figsize: Figure size
        show_stats: Whether to show statistical information
        color: Line color
        alpha: Line transparency
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series
    ax.plot(data, color=color, alpha=alpha, linewidth=1)
    
    # Set title
    if title is None:
        if H is not None:
            title = f'Fractional Gaussian Noise (fGn) - H = {H:.2f}'
        else:
            title = 'Fractional Gaussian Noise (fGn)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Show statistics
    if show_stats:
        mean_val = np.mean(data)
        std_val = np.std(data)
        stats_text = f'μ = {mean_val:.4f}\nσ = {std_val:.4f}\nN = {len(data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_fbm(data: np.ndarray, H: Optional[float] = None, title: Optional[str] = None,
             figsize: Tuple[int, int] = (12, 5), show_scaling: bool = True,
             color: str = 'darkred', alpha: float = 0.8) -> plt.Figure:
    """
    Plot fractional Brownian motion.
    
    Parameters:
        data: fBm time series
        H: Hurst exponent for title and scaling analysis
        title: Custom title
        figsize: Figure size
        show_scaling: Whether to show theoretical scaling
        color: Line color
        alpha: Line transparency
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the fBm path
    ax.plot(data, color=color, alpha=alpha, linewidth=1.5)
    
    # Set title
    if title is None:
        if H is not None:
            title = f'Fractional Brownian Motion (fBm) - H = {H:.2f}'
        else:
            title = 'Fractional Brownian Motion (fBm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Cumulative Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Show scaling properties
    if show_scaling and H is not None:
        t = np.arange(len(data))
        theoretical_std = np.sqrt(t**(2*H))
        theoretical_std = theoretical_std * np.std(data) / theoretical_std[-1]  # Normalize
        
        ax.fill_between(t, -theoretical_std, theoretical_std, 
                       alpha=0.2, color='gray', label=f'±σ(t) ∝ t^{{{H:.2f}}}')
        ax.legend()
    
    # Range information
    range_val = np.max(data) - np.min(data)
    stats_text = f'Range = {range_val:.4f}\nN = {len(data)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_fft_spectrum(freqs: np.ndarray, magnitudes: np.ndarray, 
                      title: str = 'FFT Magnitude Spectrum',
                      figsize: Tuple[int, int] = (10, 6), 
                      log_scale: bool = True, 
                      show_positive_only: bool = True) -> plt.Figure:
    """
    Plot FFT magnitude spectrum.
    
    Parameters:
        freqs: Frequency bins
        magnitudes: FFT magnitudes
        title: Plot title
        figsize: Figure size
        log_scale: Whether to use log scale on y-axis
        show_positive_only: Whether to show only positive frequencies
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_positive_only:
        # Show only positive frequencies
        n = len(freqs) // 2
        freqs_plot = freqs[:n]
        mags_plot = magnitudes[:n]
    else:
        freqs_plot = freqs
        mags_plot = magnitudes
    
    # Plot spectrum
    if log_scale:
        ax.semilogy(freqs_plot[1:], mags_plot[1:], 'b-', alpha=0.8, linewidth=1.5)  # Skip DC
        ax.set_ylabel('Magnitude (log scale)', fontsize=12)
    else:
        ax.plot(freqs_plot, mags_plot, 'b-', alpha=0.8, linewidth=1.5)
        ax.set_ylabel('Magnitude', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Highlight peak frequency
    if len(mags_plot) > 1:
        peak_idx = np.argmax(mags_plot[1:]) + 1  # Skip DC
        peak_freq = freqs_plot[peak_idx]
        peak_mag = mags_plot[peak_idx]
        ax.axvline(peak_freq, color='red', linestyle='--', alpha=0.7, 
                  label=f'Peak: f={peak_freq:.4f}')
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_wavelet_coefficients(coeffs: List[np.ndarray], wavelet: str = 'db2',
                              figsize: Tuple[int, int] = (14, 8),
                              title: Optional[str] = None) -> plt.Figure:
    """
    Plot wavelet decomposition coefficients.
    
    Parameters:
        coeffs: List of coefficient arrays from wavelet transform
        wavelet: Wavelet name for title
        figsize: Figure size
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    n_levels = len(coeffs) - 1  # Subtract 1 for approximation coefficients
    
    fig, axes = plt.subplots(n_levels + 1, 1, figsize=figsize, sharex=True)
    if n_levels == 0:  # Handle single subplot case
        axes = [axes]
    
    if title is None:
        title = f'Wavelet Decomposition ({wavelet.upper()})'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot approximation coefficients
    axes[0].plot(coeffs[0], 'b-', alpha=0.8, linewidth=1.5)
    axes[0].set_title('Approximation Coefficients', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Amplitude')
    
    # Plot detail coefficients
    colors = plt.cm.viridis(np.linspace(0, 1, n_levels))
    for i in range(n_levels):
        detail_coeffs = coeffs[i + 1]
        axes[i + 1].plot(detail_coeffs, color=colors[i], alpha=0.8, linewidth=1.5)
        axes[i + 1].set_title(f'Detail Coefficients - Level {i + 1}', fontsize=12)
        axes[i + 1].grid(True, alpha=0.3)
        axes[i + 1].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Coefficient Index')
    plt.tight_layout()
    return fig


def plot_hurst_comparison(H_values: List[float], L: int = 512, 
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Compare fGn realizations for different Hurst exponents.
    
    Parameters:
        H_values: List of Hurst exponents to compare
        L: Length of time series
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    from .core import fgn  # Import here to avoid circular import
    
    n_hurst = len(H_values)
    fig, axes = plt.subplots(n_hurst, 2, figsize=figsize)
    if n_hurst == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Comparison of Different Hurst Exponents', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_hurst))
    
    for i, H in enumerate(H_values):
        # Generate fGn and fBm
        np.random.seed(42)  # For reproducible comparison
        fgn_data = fgn(H, L)
        fbm_data = np.cumsum(np.concatenate([[0], fgn_data]))
        
        # Plot fGn
        axes[i, 0].plot(fgn_data, color=colors[i], alpha=0.8, linewidth=1)
        axes[i, 0].set_title(f'fGn: H = {H:.1f}', fontsize=12)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylabel('fGn Value')
        
        # Plot fBm
        axes[i, 1].plot(fbm_data, color=colors[i], alpha=0.8, linewidth=1.5)
        axes[i, 1].set_title(f'fBm: H = {H:.1f}', fontsize=12)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel('fBm Value')
    
    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')
    
    plt.tight_layout()
    return fig


def plot_rs_analysis(data: np.ndarray, H_true: Optional[float] = None,
                     window_sizes: Optional[np.ndarray] = None,
                     figsize: Tuple[int, int] = (10, 7)) -> plt.Figure:
    """
    Plot R/S analysis results and Hurst exponent estimation.
    
    Parameters:
        data: Time series data
        H_true: True Hurst exponent (if known)
        window_sizes: Window sizes for R/S analysis
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if window_sizes is None:
        max_window = len(data) // 4
        window_sizes = np.logspace(1, np.log10(max_window), 20).astype(int)
        window_sizes = np.unique(window_sizes)
    
    rs_values = []
    
    # Compute R/S for different window sizes
    for window_size in window_sizes:
        if window_size >= len(data):
            continue
            
        # Split data into windows
        n_windows = len(data) // window_size
        rs_window_values = []
        
        for i in range(n_windows):
            window_data = data[i*window_size:(i+1)*window_size]
            if len(window_data) < 4:
                continue
                
            mean_data = np.mean(window_data)
            cumdev = np.cumsum(window_data - mean_data)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(window_data, ddof=1)
            
            if S > 0:
                rs_window_values.append(R / S)
        
        if rs_window_values:
            rs_values.append(np.mean(rs_window_values))
        else:
            rs_values.append(np.nan)
    
    # Remove NaN values
    valid_mask = ~np.isnan(rs_values)
    window_sizes = window_sizes[:len(rs_values)][valid_mask]
    rs_values = np.array(rs_values)[valid_mask]
    
    if len(rs_values) == 0:
        raise ValueError("No valid R/S values computed")
    
    # Fit line to estimate Hurst exponent
    log_windows = np.log10(window_sizes)
    log_rs = np.log10(rs_values)
    H_estimated = np.polyfit(log_windows, log_rs, 1)[0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot R/S values
    ax.loglog(window_sizes, rs_values, 'bo-', alpha=0.7, markersize=6, linewidth=2,
              label='R/S values')
    
    # Plot fitted line
    rs_fitted = window_sizes**H_estimated * rs_values[0] / window_sizes[0]**H_estimated
    ax.loglog(window_sizes, rs_fitted, 'r--', linewidth=2,
              label=f'Fitted: H = {H_estimated:.3f}')
    
    # Plot theoretical line if H_true is provided
    if H_true is not None:
        rs_theoretical = window_sizes**H_true * rs_values[0] / window_sizes[0]**H_true
        ax.loglog(window_sizes, rs_theoretical, 'g:', linewidth=2,
                  label=f'Theoretical: H = {H_true:.3f}')
    
    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('R/S Statistic', fontsize=12)
    ax.set_title('Rescaled Range (R/S) Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with results
    if H_true is not None:
        error = abs(H_estimated - H_true)
        stats_text = f'Estimated H: {H_estimated:.3f}\nTrue H: {H_true:.3f}\nError: {error:.3f}'
    else:
        stats_text = f'Estimated H: {H_estimated:.3f}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_autocorrelation(data: np.ndarray, max_lag: Optional[int] = None,
                        H: Optional[float] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot autocorrelation function of the data using multiple methods for clarity.
    
    Parameters:
        data: Time series data
        max_lag: Maximum lag to compute
        H: Hurst exponent for theoretical comparison
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if max_lag is None:
        max_lag = min(len(data) // 4, 150)
    
    # Method 1: Statsmodels (most reliable)
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        acf_sm = sm_acf(data, nlags=max_lag, fft=True)
        lags_sm = np.arange(len(acf_sm))
        use_statsmodels = True
        print(f"Using statsmodels: first 10 values = {acf_sm[:10]}")
    except ImportError:
        use_statsmodels = False
        acf_sm = None
        lags_sm = None
    
    # Method 2: Direct numpy correlate (our current method)
    data_array = np.array(data)
    mean = np.mean(data_array)
    var = np.var(data_array)
    ndata = data_array - mean
    
    acorr_full = np.correlate(ndata, ndata, 'full')
    acf_direct = acorr_full[len(ndata)-1:]  # Take positive lags
    acf_direct = acf_direct / var / len(ndata)  # Proper normalization
    acf_direct = acf_direct[:max_lag + 1]
    lags_direct = np.arange(len(acf_direct))
    print(f"Using direct method: first 10 values = {acf_direct[:10]}")
    
    # Use the better method for plotting
    if use_statsmodels:
        acf_plot = acf_sm
        lags_plot = lags_sm
        method_name = "Statsmodels ACF"
    else:
        acf_plot = acf_direct
        lags_plot = lags_direct
        method_name = "Direct numpy.correlate"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Linear scale - show the decay clearly
    ax1.plot(lags_plot, acf_plot, 'bo-', alpha=0.8, markersize=4, linewidth=1.5,
            markerfacecolor='steelblue', label=f'Empirical ({method_name})')
    
    # Plot theoretical autocorrelation if H is provided
    if H is not None:
        theoretical_acf = np.zeros(len(lags_plot))
        for k in range(len(lags_plot)):
            if k == 0:
                theoretical_acf[k] = 1.0
            else:
                theoretical_acf[k] = 0.5 * (abs(k + 1)**(2*H) - 2*abs(k)**(2*H) + abs(k - 1)**(2*H))
        
        ax1.plot(lags_plot, theoretical_acf, 'r--', linewidth=2, alpha=0.8,
                label=f'Theoretical (H={H:.2f})')
    
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Lag', fontsize=12)
    ax1.set_ylabel('Autocorrelation', fontsize=12)
    ax1.set_title('Autocorrelation Function (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add text box showing first few values
    if H is not None and H > 0.5:
        color_status = 'lightgreen' if acf_plot[1] > 0.1 else 'lightcoral'
        status_text = 'GOOD: Shows long-range dependence' if acf_plot[1] > 0.1 else 'BAD: No long-range dependence'
    else:
        color_status = 'lightblue'
        status_text = 'Short-range dependence expected'
    
    values_text = f"""First 5 lags:
Lag 1: {acf_plot[1]:.4f}
Lag 2: {acf_plot[2]:.4f}
Lag 3: {acf_plot[3]:.4f}
Lag 4: {acf_plot[4]:.4f}
Lag 5: {acf_plot[5]:.4f}

{status_text}"""
    
    ax1.text(0.98, 0.98, values_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color_status, alpha=0.8),
            fontsize=10)
    
    # Plot 2: Log-log plot to show power law decay
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Only plot positive values for log scale
    positive_mask = acf_plot[1:] > 0
    positive_lags = lags_plot[1:][positive_mask]
    positive_acf = acf_plot[1:][positive_mask]
    
    if len(positive_acf) > 0:
        ax2.plot(positive_lags, positive_acf, 'bo-', alpha=0.8, markersize=4, 
                linewidth=1.5, markerfacecolor='steelblue', label=f'Empirical ({method_name})')
        
        # Add power law reference line
        if H is not None and len(positive_lags) > 5:
            # Use asymptotic power law: ρ(k) ~ k^(2H-2)
            ref_lags = positive_lags
            power_law = ref_lags**(2*H - 2)
            # Scale to match empirical data at lag 10 (or closest available)
            if len(positive_lags) > 5:
                scale_idx = min(5, len(positive_lags) - 1)
                scale_factor = positive_acf[scale_idx] / power_law[scale_idx]
                power_law *= scale_factor
                ax2.plot(ref_lags, power_law, 'r--', linewidth=2, alpha=0.8,
                        label=f'Power law: k^{{{2*H-2:.1f}}}')
    
    ax2.set_xlabel('Lag (log scale)', fontsize=12)
    ax2.set_ylabel('Autocorrelation (log scale)', fontsize=12)
    ax2.set_title('Log-Log Plot - Power Law Decay', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_summary(data: np.ndarray, H: Optional[float] = None, 
                figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create a comprehensive summary plot showing multiple views of the data.
    
    Parameters:
        data: fGn time series data
        H: Hurst exponent (if known)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    from .core import fbm, fft, fwt  # Import here to avoid circular import
    
    fig = plt.figure(figsize=figsize)
    
    # Create subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. fGn time series
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data, 'steelblue', alpha=0.8, linewidth=1)
    title1 = f'Fractional Gaussian Noise (H={H:.2f})' if H else 'Fractional Gaussian Noise'
    ax1.set_title(title1, fontweight='bold')
    ax1.set_ylabel('fGn Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. Statistics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""Statistics:
N = {len(data)}
μ = {np.mean(data):.4f}
σ = {np.std(data):.4f}
min = {np.min(data):.4f}
max = {np.max(data):.4f}
range = {np.ptp(data):.4f}"""
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. fBm (cumulative sum)
    ax3 = fig.add_subplot(gs[1, 0])
    fbm_data = fbm(data)
    ax3.plot(fbm_data, 'darkred', alpha=0.8, linewidth=1.5)
    ax3.set_title('Fractional Brownian Motion', fontweight='bold')
    ax3.set_ylabel('fBm Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. FFT spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    freqs, magnitudes = fft(data)
    positive_freqs = freqs[:len(freqs)//2]
    positive_mags = magnitudes[:len(magnitudes)//2]
    ax4.semilogy(positive_freqs[1:], positive_mags[1:], 'green', alpha=0.8, linewidth=1.5)
    ax4.set_title('FFT Spectrum', fontweight='bold')
    ax4.set_ylabel('Magnitude (log)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Autocorrelation (using statsmodels for reliability)
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Use statsmodels if available, otherwise fallback
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        max_lag = min(len(data) // 4, 100)
        autocorr = sm_acf(data, nlags=max_lag, fft=True)
        lags = np.arange(len(autocorr))
        method_label = "Statsmodels ACF"
        print(f"plot_summary using statsmodels ACF: first 10 = {autocorr[:10]}")
    except ImportError:
        # Fallback to direct method
        max_lag = min(len(data) // 4, 100)
        data_array = np.array(data)
        mean = np.mean(data_array)
        var = np.var(data_array)
        ndata = data_array - mean
        
        acorr_full = np.correlate(ndata, ndata, 'full')
        autocorr = acorr_full[len(ndata)-1:]  # Take positive lags
        autocorr = autocorr / var / len(ndata)  # Proper normalization
        autocorr = autocorr[:max_lag + 1]
        lags = np.arange(len(autocorr))
        method_label = "Direct method"
        print(f"plot_summary using direct method: first 10 = {autocorr[:10]}")
    
    # Plot autocorrelation with clear markers
    ax5.plot(lags, autocorr, 'purple', '-', alpha=0.8, linewidth=2, label=method_label)
    ax5.plot(lags[::5], autocorr[::5], 'purple', 'o', alpha=0.9, markersize=4)  # Show every 5th point clearly
    
    # Add status indicator
    if H is not None and H > 0.5:
        if autocorr[1] > 0.1:
            status_text = f"✓ SLOW DECAY (H={H:.1f})"
            title_color = 'green'
        else:
            status_text = f"✗ FAST DECAY (H={H:.1f})"
            title_color = 'red'
    else:
        status_text = f"Expected fast decay (H={H:.1f})" if H else "Autocorrelation"
        title_color = 'blue'
    
    ax5.set_title(f'Autocorrelation\n{status_text}', fontweight='bold', color=title_color, fontsize=10)
    ax5.set_ylabel('ρ(lag)')
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.grid(True, alpha=0.3)
    
    # Add text showing key values
    if len(autocorr) > 10:
        values_text = f"ρ(1)={autocorr[1]:.3f}\nρ(5)={autocorr[5]:.3f}\nρ(10)={autocorr[10]:.3f}"
        ax5.text(0.02, 0.98, values_text, transform=ax5.transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                fontsize=9)
    
    # 6. Histogram
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(data, bins=30, alpha=0.7, color='orange', edgecolor='black', density=True)
    # Overlay normal distribution
    x_norm = np.linspace(np.min(data), np.max(data), 100)
    y_norm = (1/np.sqrt(2*np.pi*np.var(data))) * np.exp(-0.5*(x_norm - np.mean(data))**2/np.var(data))
    ax6.plot(x_norm, y_norm, 'r--', linewidth=2, label='Normal fit')
    ax6.set_title('Distribution', fontweight='bold')
    ax6.set_ylabel('Density')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Wavelet coefficients (approximation only)
    ax7 = fig.add_subplot(gs[2, 1])
    coeffs = fwt(data)
    ax7.plot(coeffs[0], 'brown', alpha=0.8, linewidth=1)
    ax7.set_title('Wavelet Approximation', fontweight='bold')
    ax7.set_ylabel('Coefficient')
    ax7.grid(True, alpha=0.3)
    
    # 8. R/S analysis (simplified)
    ax8 = fig.add_subplot(gs[2, 2])
    try:
        # Quick R/S analysis for a few window sizes
        window_sizes = [16, 32, 64, 128]
        rs_values = []
        for ws in window_sizes:
            if ws >= len(data):
                continue
            window_data = data[:ws]
            mean_data = np.mean(window_data)
            cumdev = np.cumsum(window_data - mean_data)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(window_data, ddof=1)
            if S > 0:
                rs_values.append(R / S)
        
        if len(rs_values) > 1:
            valid_ws = window_sizes[:len(rs_values)]
            ax8.loglog(valid_ws, rs_values, 'mo-', alpha=0.8, markersize=6)
            # Fit and show estimated H
            if len(rs_values) >= 2:
                H_est = np.polyfit(np.log(valid_ws), np.log(rs_values), 1)[0]
                ax8.set_title(f'R/S Analysis (H≈{H_est:.2f})', fontweight='bold')
            else:
                ax8.set_title('R/S Analysis', fontweight='bold')
        else:
            ax8.text(0.5, 0.5, 'Insufficient data\nfor R/S analysis', 
                    transform=ax8.transAxes, ha='center', va='center')
            ax8.set_title('R/S Analysis', fontweight='bold')
    except:
        ax8.text(0.5, 0.5, 'R/S analysis\nfailed', 
                transform=ax8.transAxes, ha='center', va='center')
        ax8.set_title('R/S Analysis', fontweight='bold')
    
    ax8.set_ylabel('R/S')
    ax8.grid(True, alpha=0.3)
    
    # Set common x-labels for bottom row
    ax6.set_xlabel('Value')
    ax7.set_xlabel('Coefficient Index')
    ax8.set_xlabel('Window Size')
    
    # Overall title
    main_title = f'FractalSig Analysis Summary (H={H:.2f})' if H else 'FractalSig Analysis Summary'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    return fig 