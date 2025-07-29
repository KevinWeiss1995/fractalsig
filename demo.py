#!/usr/bin/env python3
"""
Advanced demonstration of the FractalSig library capabilities.
Showcases all helper functions for analysis, plotting, and utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from fractalsig import (
    # Core functions
    fgn, fbm, fft, fwt,
    # Analysis functions
    rs_analysis, dfa_analysis, estimate_hurst_multiple_methods,
    autocorrelation_function, validate_fgn_properties,
    # Plotting functions
    plot_fgn, plot_fbm, plot_fft_spectrum, plot_hurst_comparison,
    plot_rs_analysis, plot_autocorrelation, plot_summary,
    # Utility functions
    benchmark_fgn_methods, validate_algorithm_correctness,
    generate_test_dataset, create_report, get_system_info
)


def demo_advanced_analysis():
    """Demonstrate advanced analysis capabilities."""
    print("=== ADVANCED ANALYSIS DEMO ===\n")
    
    # Generate test data
    np.random.seed(42)
    H_true = 0.7
    L = 1024
    
    print(f"1. Generating fGn with H={H_true}, L={L}")
    fgn_data = fgn(H_true, L)
    print(f"   Generated series: mean={np.mean(fgn_data):.4f}, std={np.std(fgn_data):.4f}")
    
    # Multiple Hurst estimation methods
    print(f"\n2. Multiple Hurst Estimation Methods:")
    hurst_results = estimate_hurst_multiple_methods(fgn_data)
    
    for method, result in hurst_results.items():
        if method == 'summary':
            continue
        if 'H' in result:
            error = abs(result['H'] - H_true)
            print(f"   {result['method']}: H = {result['H']:.4f} (error: {error:.4f})")
        else:
            print(f"   {result.get('method', method)}: Failed - {result.get('error', 'Unknown error')}")
    
    if 'summary' in hurst_results:
        summary = hurst_results['summary']
        print(f"   Summary: Mean H = {summary['mean_H']:.4f} ± {summary['std_H']:.4f}")
    
    # R/S Analysis detailed
    print(f"\n3. Detailed R/S Analysis:")
    H_rs, window_sizes, rs_values = rs_analysis(fgn_data)
    print(f"   Estimated H = {H_rs:.4f}")
    print(f"   Window sizes tested: {len(window_sizes)} ranging from {min(window_sizes)} to {max(window_sizes)}")
    
    # DFA Analysis
    print(f"\n4. Detrended Fluctuation Analysis (DFA):")
    try:
        H_dfa, dfa_windows, fluctuations = dfa_analysis(fgn_data)
        print(f"   Estimated H = {H_dfa:.4f}")
        print(f"   Scaling range: {min(dfa_windows)} to {max(dfa_windows)}")
    except Exception as e:
        print(f"   DFA failed: {str(e)}")
    
    # Validation
    print(f"\n5. Statistical Validation:")
    validation = validate_fgn_properties(fgn_data, H_true)
    
    if 'normality' in validation:
        norm_result = validation['normality']
        print(f"   Normality test ({norm_result['test']}): {'PASS' if norm_result['is_normal'] else 'FAIL'}")
    
    if 'hurst_validation' in validation:
        hurst_val = validation['hurst_validation']
        if 'error' not in hurst_val:
            print(f"   Hurst accuracy: {'PASS' if hurst_val['is_accurate'] else 'FAIL'} (error: {hurst_val['error']:.4f})")
    
    return fgn_data, H_true


def demo_advanced_plotting():
    """Demonstrate advanced plotting capabilities."""
    print("\n=== ADVANCED PLOTTING DEMO ===\n")
    
    np.random.seed(42)
    
    # Generate data for multiple Hurst values
    H_values = [0.3, 0.5, 0.7, 0.9]
    L = 256
    
    print("1. Creating Hurst Comparison Plot")
    fig1 = plot_hurst_comparison(H_values, L=L, figsize=(16, 12))
    fig1.savefig('hurst_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: hurst_comparison.png")
    
    # Detailed analysis of one series
    fgn_data = fgn(0.7, 512)
    
    print("\n2. Creating Individual Analysis Plots")
    
    # fGn plot
    fig2 = plot_fgn(fgn_data, H=0.7, show_stats=True)
    fig2.savefig('fgn_detailed.png', dpi=150, bbox_inches='tight')
    print("   Saved: fgn_detailed.png")
    
    # fBm plot
    fbm_data = fbm(fgn_data)
    fig3 = plot_fbm(fbm_data, H=0.7, show_scaling=True)
    fig3.savefig('fbm_detailed.png', dpi=150, bbox_inches='tight')
    print("   Saved: fbm_detailed.png")
    
    # FFT spectrum
    freqs, magnitudes = fft(fgn_data)
    fig4 = plot_fft_spectrum(freqs, magnitudes, log_scale=True)
    fig4.savefig('fft_spectrum.png', dpi=150, bbox_inches='tight')
    print("   Saved: fft_spectrum.png")
    
    # R/S analysis plot
    try:
        fig5 = plot_rs_analysis(fgn_data, H_true=0.7)
        fig5.savefig('rs_analysis.png', dpi=150, bbox_inches='tight')
        print("   Saved: rs_analysis.png")
    except Exception as e:
        print(f"   R/S plot failed: {str(e)}")
    
    # Autocorrelation plot
    fig6 = plot_autocorrelation(fgn_data, H=0.7)
    fig6.savefig('autocorrelation.png', dpi=150, bbox_inches='tight')
    print("   Saved: autocorrelation.png")
    
    # Comprehensive summary
    print("\n3. Creating Comprehensive Summary Plot")
    fig7 = plot_summary(fgn_data, H=0.7)
    fig7.savefig('comprehensive_summary.png', dpi=150, bbox_inches='tight')
    print("   Saved: comprehensive_summary.png")
    
    plt.close('all')  # Close all figures to save memory


def demo_utility_functions():
    """Demonstrate utility and benchmarking functions."""
    print("\n=== UTILITY FUNCTIONS DEMO ===\n")
    
    # System information
    print("1. System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"   {key}: {value}")
    
    # Algorithm validation
    print("\n2. Algorithm Validation:")
    validation_results = validate_algorithm_correctness(n_tests=5, seed=42)
    
    for test_name, passed in validation_results.items():
        if test_name == 'error_messages':
            continue
        status = "PASS" if passed else "FAIL"
        print(f"   {test_name}: {status}")
    
    if not validation_results['all_tests_passed'] and validation_results['error_messages']:
        print("   Errors:")
        for error in validation_results['error_messages']:
            print(f"     - {error}")
    
    # Benchmarking
    print("\n3. Performance Benchmarking:")
    H_values = [0.5, 0.7]
    L_values = [128, 256, 512]
    
    print("   Running benchmarks (this may take a moment)...")
    benchmark_results = benchmark_fgn_methods(H_values, L_values, n_trials=3)
    
    print("\n   Timing Results (seconds):")
    for H in H_values:
        print(f"   H = {H}:")
        for L in L_values:
            if L in benchmark_results['timings'][H]:
                timing = benchmark_results['timings'][H][L]
                print(f"     L={L}: {timing['mean_time']:.4f} ± {timing['std_time']:.4f}")
    
    # Test dataset generation
    print("\n4. Test Dataset Generation:")
    test_data = generate_test_dataset(
        H=0.6, L=256, 
        noise_level=0.1, 
        trend='linear',
        seasonal_period=50
    )
    
    print(f"   Generated dataset with {len(test_data)} components:")
    for component, data in test_data.items():
        if isinstance(data, np.ndarray):
            print(f"     {component}: shape {data.shape}, range [{np.min(data):.3f}, {np.max(data):.3f}]")
        else:
            print(f"     {component}: {data}")
    
    return test_data


def demo_comprehensive_report():
    """Demonstrate comprehensive reporting capabilities."""
    print("\n=== COMPREHENSIVE REPORTING DEMO ===\n")
    
    np.random.seed(42)
    
    # Generate data
    H_true = 0.75
    L = 512
    fgn_data = fgn(H_true, L)
    
    print(f"1. Generating comprehensive report for H={H_true}, L={L}")
    
    # Create report
    report = create_report(fgn_data, H=H_true, output_file='fractalsig_report.txt')
    
    print("2. Report generated and saved to: fractalsig_report.txt")
    print("\n3. Report preview (first 500 characters):")
    print("-" * 60)
    print(report[:500] + "..." if len(report) > 500 else report)
    print("-" * 60)
    
    # Report summary
    lines = report.split('\n')
    sections = [line for line in lines if line.startswith('BASIC STATISTICS') or 
                line.startswith('HURST EXPONENT') or line.startswith('VALIDATION') or
                line.startswith('SPECTRAL')]
    
    print(f"\n4. Report contains {len(sections)} main sections:")
    for section in sections:
        print(f"   - {section}")


def demo_comparison_analysis():
    """Demonstrate time series comparison capabilities."""
    print("\n=== COMPARISON ANALYSIS DEMO ===\n")
    
    np.random.seed(42)
    
    # Generate two different series
    H1, H2 = 0.3, 0.8
    L = 256
    
    print(f"1. Comparing two fGn series: H1={H1}, H2={H2}")
    
    fgn1 = fgn(H1, L)
    fgn2 = fgn(H2, L)
    
    # Compare using utility function
    from fractalsig.analysis import compare_time_series
    comparison = compare_time_series(fgn1, fgn2, labels=(f'fGn (H={H1})', f'fGn (H={H2})'))
    
    print("2. Comparison Results:")
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Visual comparison
    print("\n3. Creating Visual Comparison")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Comparison: H={H1} vs H={H2}', fontsize=14, fontweight='bold')
    
    # Time series plots
    axes[0,0].plot(fgn1, 'b-', alpha=0.7, label=f'H={H1}')
    axes[0,0].plot(fgn2, 'r-', alpha=0.7, label=f'H={H2}')
    axes[0,0].set_title('Time Series')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Histograms
    axes[0,1].hist(fgn1, bins=30, alpha=0.7, label=f'H={H1}', density=True, color='blue')
    axes[0,1].hist(fgn2, bins=30, alpha=0.7, label=f'H={H2}', density=True, color='red')
    axes[0,1].set_title('Distributions')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # FFT comparison
    freqs1, mags1 = fft(fgn1)
    freqs2, mags2 = fft(fgn2)
    pos_freqs = freqs1[:len(freqs1)//2]
    axes[1,0].semilogy(pos_freqs[1:], mags1[:len(mags1)//2][1:], 'b-', alpha=0.7, label=f'H={H1}')
    axes[1,0].semilogy(pos_freqs[1:], mags2[:len(mags2)//2][1:], 'r-', alpha=0.7, label=f'H={H2}')
    axes[1,0].set_title('FFT Spectra')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Autocorrelation comparison
    lags1, autocorr1 = autocorrelation_function(fgn1, max_lag=50)
    lags2, autocorr2 = autocorrelation_function(fgn2, max_lag=50)
    axes[1,1].plot(lags1, autocorr1, 'b-o', alpha=0.7, markersize=3, label=f'H={H1}')
    axes[1,1].plot(lags2, autocorr2, 'r-o', alpha=0.7, markersize=3, label=f'H={H2}')
    axes[1,1].set_title('Autocorrelation')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved: comparison_analysis.png")
    plt.close()


def main():
    """Run the complete advanced demo."""
    print("🚀 FractalSig Advanced Features Demonstration")
    print("=" * 60)
    
    try:
        # Run all demo sections
        fgn_data, H_true = demo_advanced_analysis()
        demo_advanced_plotting()
        test_data = demo_utility_functions()
        demo_comprehensive_report()
        demo_comparison_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ADVANCED DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nGenerated Files:")
        files_created = [
            'hurst_comparison.png',
            'fgn_detailed.png', 
            'fbm_detailed.png',
            'fft_spectrum.png',
            'rs_analysis.png',
            'autocorrelation.png',
            'comprehensive_summary.png',
            'comparison_analysis.png',
            'fractalsig_report.txt'
        ]
        
        for i, filename in enumerate(files_created, 1):
            print(f"{i:2d}. {filename}")
        
        print("\nKey Capabilities Demonstrated:")
        capabilities = [
            "✓ Multiple Hurst estimation methods (R/S, DFA, Wavelet)",
            "✓ Advanced statistical validation",
            "✓ Comprehensive visualization suite", 
            "✓ Performance benchmarking",
            "✓ Algorithm correctness validation",
            "✓ Test dataset generation",
            "✓ Detailed analysis reporting",
            "✓ Time series comparison tools",
            "✓ System information and debugging"
        ]
        
        for capability in capabilities:
            print(f"  {capability}")
            
        print(f"\n📊 Analysis Summary:")
        print(f"  • Original fGn: H={H_true}, length={len(fgn_data)}")
        print(f"  • Multiple estimation methods tested")
        print(f"  • {len(files_created)} visualization files created")
        print(f"  • Comprehensive report generated")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        plt.close('all')  # Ensure all plots are closed


if __name__ == "__main__":
    main() 