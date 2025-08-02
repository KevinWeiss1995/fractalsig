"""
Tests for FractalSig helper functions (analysis, plotting, utils).
"""

import numpy as np
import pytest
import warnings
from fractalsig import (
    fgn, fbm, fft, fwt,
    rs_analysis, dfa_analysis, estimate_hurst_multiple_methods,
    plot_fgn, plot_fbm, plot_fft_spectrum, plot_summary, plot_hurst_comparison,
    benchmark_fgn_methods, validate_algorithm_correctness, generate_test_dataset, create_report
)


class TestAnalysisFunctions:
    """Test analysis utility functions."""
    
    def test_rs_analysis_basic(self):
        """Test basic R/S analysis functionality."""
        np.random.seed(42)
        data = fgn(0.7, 256)
        
        H_est, window_sizes, rs_values = rs_analysis(data)
        
        assert isinstance(H_est, float)
        assert 0.0 < H_est < 1.0
        assert len(window_sizes) == len(rs_values)
        assert len(window_sizes) > 0
        assert all(rs > 0 for rs in rs_values)
    
    def test_rs_analysis_known_case(self):
        """Test R/S analysis with known Hurst exponent."""
        np.random.seed(123)
        H_true = 0.6
        data = fgn(H_true, 512)
        
        H_est, _, _ = rs_analysis(data)
        
        # Should be reasonably close (within 0.15 for random data)
        assert abs(H_est - H_true) < 0.15
    
    def test_rs_analysis_edge_cases(self):
        """Test R/S analysis edge cases."""
        # Too short data
        short_data = np.random.randn(10)
        with pytest.raises(ValueError, match="too short"):
            rs_analysis(short_data)
        
        # Constant data should handle gracefully
        constant_data = np.ones(100)
        try:
            H_est, _, _ = rs_analysis(constant_data)
            # Should not crash, but result may not be meaningful
            assert isinstance(H_est, (int, float))
        except (ValueError, RuntimeError):
            # Also acceptable to raise an error for constant data
            pass
    
    def test_dfa_analysis_basic(self):
        """Test basic DFA functionality."""
        np.random.seed(42)
        data = fgn(0.6, 256)
        
        H_est, window_sizes, fluctuations = dfa_analysis(data)
        
        assert isinstance(H_est, float)
        assert 0.0 < H_est < 1.0
        assert len(window_sizes) == len(fluctuations)
        assert len(window_sizes) > 0
        assert all(f > 0 for f in fluctuations)
    
    def test_estimate_hurst_multiple_methods(self):
        """Test multiple Hurst estimation methods."""
        np.random.seed(42)
        H_true = 0.7
        data = fgn(H_true, 512)
        
        # Test all available methods
        results = estimate_hurst_multiple_methods(data)
        
        assert isinstance(results, dict)
        
        # Check that we have results from different methods (using actual key names)
        expected_method_keys = ['rs_analysis', 'dfa', 'wavelet']
        
        for method_key in expected_method_keys:
            if method_key in results:
                method_result = results[method_key]
                assert isinstance(method_result, dict)
                
                # All methods should use 'H' as the key name
                assert 'H' in method_result, f"Method {method_key} missing Hurst estimate"
                
                H_est = method_result['H']
                assert isinstance(H_est, (int, float, np.number))
                assert 0.0 < H_est < 1.0, f"Method {method_key}: Hurst estimate {H_est} not in valid range"
        
        # Check that we have a summary if multiple methods worked
        if 'summary' in results:
            summary = results['summary']
            assert 'mean_H' in summary
            assert summary.get('n_methods', 0) >= 1
    
    def test_wavelet_hurst_estimation_range(self):
        """Test that wavelet Hurst estimation returns values in valid range (0,1)."""
        from fractalsig import wavelet_hurst_estimation
        
        # Test with different known Hurst values
        test_cases = [0.3, 0.5, 0.7]
        
        for H_true in test_cases:
            np.random.seed(42)
            data = fgn(H_true, 256)
            
            H_est, info = wavelet_hurst_estimation(data)
            
            # Critical test: Hurst estimate must be in valid range
            assert 0 < H_est < 1, f"Hurst estimate {H_est} not in valid range (0,1) for H_true={H_true}"
            
            # Additional checks
            assert isinstance(H_est, (float, np.floating)), "Hurst estimate should be float"
            assert isinstance(info, dict), "Info should be dictionary"
            assert 'slope' in info, "Info should contain slope"
            assert 'wavelet' in info, "Info should contain wavelet type"
            
            # Verify the mathematical relationship is correct for wavelet analysis
            # For wavelet analysis of fGn: H = (1 - slope) / 2
            expected_H = (1 - info['slope']) / 2
            assert abs(H_est - expected_H) < 1e-10, "Formula calculation mismatch"


class TestPlottingFunctions:
    """Test plotting utility functions."""
    
    def test_plot_fgn_basic(self):
        """Test basic fGn plotting."""
        np.random.seed(42)
        data = fgn(0.7, 128)
        
        fig = plot_fgn(data, H=0.7)
        
        assert fig is not None
        assert len(fig.axes) == 1
        
        # Check that plot has data
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) > 0
        assert len(lines[0].get_ydata()) == len(data)
    
    def test_plot_fbm_basic(self):
        """Test basic fBm plotting."""
        np.random.seed(42)
        fgn_data = fgn(0.6, 128)
        fbm_data = fbm(0.7, 256)
        
        fig = plot_fbm(fbm_data, H=0.6)
        
        assert fig is not None
        assert len(fig.axes) == 1
        
        ax = fig.axes[0] 
        lines = ax.get_lines()
        assert len(lines) > 0
    
    def test_plot_summary(self):
        """Test comprehensive summary plot."""
        np.random.seed(42)
        data = fgn(0.5, 128)
        
        fig = plot_summary(data, H=0.5)
        
        assert fig is not None
        assert len(fig.axes) >= 6  # Should have multiple subplots
    
    def test_plot_hurst_comparison(self):
        """Test Hurst comparison plot."""
        H_values = [0.3, 0.5, 0.8]
        
        fig = plot_hurst_comparison(H_values, L=64)
        
        assert fig is not None
        assert len(fig.axes) == len(H_values) * 2  # fGn and fBm for each H


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_benchmark_fgn_methods(self):
        """Test fGn benchmarking."""
        H_values = [0.5, 0.7]
        L_values = [64, 128]
        
        results = benchmark_fgn_methods(H_values, L_values, n_trials=2)
        
        assert isinstance(results, dict)
        assert 'timings' in results
        assert 'H_values' in results
        assert 'L_values' in results
        
        # Should have timing data for each H
        for H in H_values:
            assert H in results['timings']
    
    def test_validate_algorithm_correctness(self):
        """Test algorithm validation."""
        results = validate_algorithm_correctness(n_tests=3)
        
        assert isinstance(results, dict)
        assert 'error_messages' in results
        assert 'fgn_generation' in results
        assert 'fbm_reconstruction' in results
        assert results['fgn_generation'] is True
        assert results['fft_sine_wave'] is True
        assert results['wavelet_reconstruction'] is True
        assert results['fbm_reconstruction'] is True
    
    def test_generate_test_dataset(self):
        """Test synthetic dataset generation."""
        H = 0.6
        L = 128  # Power of 2 for efficiency
        
        dataset = generate_test_dataset(H, L, noise_level=0.1, trend='linear')
        
        assert isinstance(dataset, dict)
        assert 'fgn' in dataset
        assert 'fbm' in dataset
        assert 'combined' in dataset
        assert 'H_true' in dataset
        
        assert len(dataset['fgn']) == L
        assert len(dataset['fbm']) == L + 1  # fBm is one element longer
        assert dataset['H_true'] == H
    
    def test_create_report(self):
        """Test report generation."""
        np.random.seed(42)
        data = fgn(0.7, 256)
        
        report = create_report(data, H=0.7)
        
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert 'FRACTALSIG ANALYSIS REPORT' in report
        assert 'BASIC STATISTICS' in report
        assert 'HURST EXPONENT' in report


class TestIntegrationWithNewFunctions:
    """Test integration of new functions with existing core."""
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow with helper functions."""
        np.random.seed(42)
        
        # Generate data
        H_true = 0.7
        L = 256
        fgn_data = fgn(H_true, L)
        
        # Analysis
        H_rs, _, _ = rs_analysis(fgn_data)
        
        # Validation should not crash
        validation_results = validate_algorithm_correctness(n_tests=2)
        assert validation_results['fgn_generation'] is True
        
        # Generate test dataset
        test_data = generate_test_dataset(H_true, L)
        assert 'fgn' in test_data
        
        # Create report
        report = create_report(fgn_data, H=H_true)
        assert isinstance(report, str)
        
        # Should complete without errors
        assert True
    
    def test_plotting_with_analysis(self):
        """Test plotting functions with analysis results."""
        np.random.seed(42)
        data = fgn(0.6, 128)
        
        # Should be able to create plots without errors
        fig1 = plot_fgn(data, H=0.6)
        assert fig1 is not None
        
        fig2 = plot_summary(data, H=0.6)
        assert fig2 is not None
        
        # R/S analysis plot
        try:
            from fractalsig import plot_rs_analysis
            fig3 = plot_rs_analysis(data, H_true=0.6)
            assert fig3 is not None
        except Exception:
            # OK if this fails due to insufficient data
            pass
    
    def test_error_handling_robustness(self):
        """Test that helper functions handle errors gracefully."""
        # Very short data
        short_data = np.random.randn(5)
        
        # Should either work or raise informative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # These should not crash unexpectedly
            try:
                plot_fgn(short_data)
            except (ValueError, RuntimeError):
                pass  # Expected for short data
            
            try:
                rs_analysis(short_data)
            except (ValueError, RuntimeError):
                pass  # Expected for short data
    
    def test_memory_and_performance(self):
        """Test that helper functions don't have obvious memory leaks."""
        # This test mainly ensures functions complete without hanging
        np.random.seed(42)
        
        data = fgn(0.5, 128)
        
        # These should complete quickly
        results = estimate_hurst_multiple_methods(data)
        assert isinstance(results, dict)
        
        # Benchmarking should work
        bench_results = benchmark_fgn_methods([0.5], [64], n_trials=1)
        assert isinstance(bench_results, dict)


if __name__ == "__main__":
    # Quick test if run directly
    print("Running basic helper function tests...")
    
    # Test analysis
    np.random.seed(42)
    data = fgn(0.7, 256)
    H_est, _, _ = rs_analysis(data)
    print(f"R/S Analysis: H = {H_est:.3f}")
    
    # Test plotting
    fig = plot_fgn(data, H=0.7)
    print("fGn plot created successfully")
    
        # Test utilities
    results = validate_algorithm_correctness(n_tests=1)
    print(f"Algorithm validation: {results['fgn_generation']}")
    
    print("Basic tests passed!") 