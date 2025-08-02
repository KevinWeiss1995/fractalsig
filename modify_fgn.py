#!/usr/bin/env python3
"""
Script to modify the fgn function to use Cholesky as primary method.
"""

def modify_fgn_function():
    # Read the current file
    with open('fractalsig/core.py', 'r') as f:
        content = f.read()
    
    # Find the fgn function and replace it
    old_function_start = "def fgn(H, L):"
    old_function_end = "    return fgn_series"
    
    # Find the start and end positions
    start_pos = content.find(old_function_start)
    if start_pos == -1:
        print("ERROR: Could not find fgn function")
        return False
    
    # Find the end of the function (look for the return statement)
    temp_pos = start_pos
    end_pos = -1
    while temp_pos < len(content):
        line_start = content.find('\n', temp_pos) + 1
        if line_start == 0:  # No more newlines
            break
        line_end = content.find('\n', line_start)
        if line_end == -1:
            line_end = len(content)
        
        line = content[line_start:line_end]
        if line.strip() == "return fgn_series":
            end_pos = line_end
            break
        temp_pos = line_end
    
    if end_pos == -1:
        print("ERROR: Could not find end of fgn function")
        return False
    
    # Create the new function
    new_function = '''def fgn(H, L):
    """
    Generate a fractional Gaussian noise (fGn) time series.
    
    Uses Cholesky decomposition for L < 2^20 (more stable, exact).
    Uses Davies-Harte method for L >= 2^20 (more efficient for large datasets).
    
    Parameters:
        H (float): Hurst exponent (0 < H < 1)
        L (int): Length of the time series
        
    Returns:
        np.ndarray: Array of shape (L,) representing fractional Gaussian noise
        
    Raises:
        ValueError: If H is not in (0, 1)
        UserWarning: If L is not a power of two (for Davies-Harte method)
    """
    if not (0 < H < 1):
        raise ValueError(f"Hurst exponent H must be in (0, 1), got {H}")
    
    if L <= 0:
        raise ValueError(f"Length L must be positive, got {L}")
    
    # Handle edge case for very small lengths
    if L == 1:
        return np.array([np.random.randn()])
    
    # Use Cholesky method as primary approach for L < 2^20 (1,048,576)
    # This is more numerically stable and exact
    if L < 2**20:
        return _fgn_simple(H, L)
    
    # For very large datasets (L >= 2^20), use Davies-Harte for efficiency
    # Check if L is a power of two and warn if not
    if L > 1 and (L & (L - 1)) != 0:
        import warnings
        # Find the next power of two for suggestion
        next_power_of_two = 1 << (L - 1).bit_length()
        prev_power_of_two = next_power_of_two >> 1
        
        warnings.warn(
            f"Length L={L} is not a power of two. The Davies-Harte method uses FFT operations "
            f"which are most efficient with power-of-two lengths (e.g., {prev_power_of_two}, {next_power_of_two}). "
            f"Non-power-of-two lengths will result in slower FFT computations and may cause "
            f"the circulant embedding matrix to have a larger size (2*(L-1)={2*(L-1)}), "
            f"potentially leading to increased memory usage and computation time. "
            f"Consider using a nearby power of two for optimal performance.",
            UserWarning,
            stacklevel=2
        )
    
    # Davies-Harte method for large datasets
    # Reference: Wood & Chan (1994), Dietrich & Newsam (1997)
    
    # Step 1: Compute autocovariance sequence γ(k) for fGn
    # Using exact formula: γ(k) = 0.5 * [|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H)]
    gamma = np.zeros(L)
    gamma[0] = 1.0  # γ(0) = variance = 1
    
    for k in range(1, L):
        gamma[k] = 0.5 * ((k + 1)**(2*H) - 2*k**(2*H) + (k - 1)**(2*H))
    
    # Step 2: Construct circulant embedding matrix
    # Use size m = 2*L to ensure positive definiteness
    m = 2 * L
    c = np.zeros(m)
    
    # First row of circulant matrix: [γ(0), γ(1), ..., γ(L-1), 0, γ(L-1), ..., γ(1)]
    c[:L] = gamma                    # γ(0) to γ(L-1)
    c[L] = 0                        # Critical: zero at position L
    c[L+1:] = gamma[L-1:0:-1]       # γ(L-1) down to γ(1) in reverse
    
    # Step 3: Compute eigenvalues with FFT
    eigenvals = np.real(scipy_fft(c))
    
    # Quick check - if Davies-Harte fails, fall back to Cholesky
    if np.any(eigenvals < -1e-12):
        return _fgn_simple(H, L)

    eigenvals = np.maximum(eigenvals, 0)
    
    # Step 4: Generate Hermitian symmetric complex Gaussian noise
    
    # Generate independent standard normal variables
    u = np.random.randn(m)
    v = np.random.randn(m)
    
    # Create complex noise with Hermitian symmetry
    Z = np.zeros(m, dtype=complex)
    
    # DC component (k=0) must be real
    Z[0] = u[0]
    
    # Nyquist component (k=m/2) must be real if m is even
    if m % 2 == 0:
        Z[m // 2] = u[m // 2]
        # Fill positive frequencies with conjugate pairs
        for k in range(1, m // 2):
            Z[k] = (u[k] + 1j * v[k]) / np.sqrt(2)
            Z[m - k] = (u[k] - 1j * v[k]) / np.sqrt(2)  # Complex conjugate
    else:
        # For odd m, no Nyquist frequency
        for k in range(1, (m + 1) // 2):
            Z[k] = (u[k] + 1j * v[k]) / np.sqrt(2)
            Z[m - k] = (u[k] - 1j * v[k]) / np.sqrt(2)  # Complex conjugate
    
    # Step 5: Scale by square root of eigenvalues
    W = Z * np.sqrt(eigenvals)
    
    # Step 6: Inverse FFT to get the result
    X = scipy_ifft(W)
    
    # Extract first L values and take real part
    fgn_series = np.real(X[:L])
    
    # Step 7: Normalize to ensure unit variance
    actual_var = np.var(fgn_series)
    if actual_var > 1e-12:  # Avoid division by zero
        fgn_series = fgn_series / np.sqrt(actual_var)
    
    return fgn_series'''
    
    # Replace the function
    new_content = content[:start_pos] + new_function + content[end_pos:]
    
    # Write back to file
    with open('fractalsig/core.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully modified fgn function!")
    print("   - Primary method: Cholesky (_fgn_simple) for L < 2^20")
    print("   - Fallback method: Davies-Harte for L >= 2^20")
    print("   - This should eliminate wild H estimates in validation")
    
    return True

if __name__ == "__main__":
    modify_fgn_function()
