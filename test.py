import numpy as np
import cmath

# Core recursive Stockham FFT function
def fft0_py_corrected(n: int, s: int, eo: bool, x: np.ndarray, y: np.ndarray):
    """
    Recursive core of the Stockham FFT.

    n (int): Current size of the FFT block.
    s (int): Stride for accessing elements.
    eo (bool): Flag indicating buffer roles. If True, y becomes the output buffer
               and x becomes the input buffer for the *next* recursive stage.
               Also used in the base case to determine if a copy is needed.
               For the current stage (in this function call), x is the output, y is the input.
    x (np.ndarray): The NumPy array designated as the output buffer for the current stage.
    y (np.ndarray): The NumPy array designated as the input buffer for the current stage.
    """
    m = n // 2
    theta0 = 2 * np.pi / n

    if n == 1:
        # Base case: If eo is True, it implies that 'x' (the current output buffer)
        # needs to be populated from 'y' (the current input buffer) because
        # 'y' holds the actual data for this 1-point FFT.
        # If eo is False, 'x' already contains the correct value.
        if eo:
            x[:s] = y[:s]
        return
    else:
        # Recursive step: The buffers' roles are swapped for the next level.
        # The current 'y' becomes the output buffer for the sub-problem,
        # and the current 'x' becomes the input buffer for the sub-problem.
        # The 'eo' flag is toggled for the next recursion depth.
        fft0_py_corrected(n // 2, 2 * s, not eo, y, x)

        # Butterfly operation: Reads from y (input for current stage), writes to x (output for current stage)
        for p in range(m):
            # Twiddle factor: exp(-2*pi*i*p/n)
            wp = cmath.cos(p * theta0) - 1j * cmath.sin(p * theta0)
            for q in range(s):
                # Calculate indices for reading from y (input)
                idx_y0 = q + s * (2 * p + 0)
                idx_y1 = q + s * (2 * p + 1)

                # Calculate indices for writing to x (output)
                idx_x0 = q + s * (p + 0)
                idx_x1 = q + s * (p + m)

                a = y[idx_y0]
                b = y[idx_y1] * wp

                x[idx_x0] = a + b
                x[idx_x1] = a - b

# Forward FFT wrapper
def fft_py(x_in: np.ndarray) -> np.ndarray:
    """
    Performs the forward FFT using the Stockham algorithm.
    The result is scaled by 1/N, as per the original C++ code's `fft` function.

    x_in (np.ndarray): Input signal (complex numbers).
    Returns (np.ndarray): The FFT of the input signal, scaled by 1/N.
    """
    n = len(x_in)
    # 'x_out' will be the final output buffer, initially empty (zeros).
    x_out = np.zeros(n, dtype=np.complex128)
    # 'y_in' will be the initial input buffer for fft0, loaded with the actual data.
    y_in = np.array(x_in, dtype=np.complex128)

    # Initial call to fft0_py_corrected:
    # x_out is the designated output buffer, y_in is the designated input buffer (holding the data).
    # 'eo' starts as False.
    fft0_py_corrected(n, 1, False, x_out, y_in)

    # Apply final scaling as per the C++ `fft` function
    return x_out / n

# Inverse FFT wrapper
def ifft_py(x_in: np.ndarray) -> np.ndarray:
    """
    Performs the inverse FFT using the Stockham algorithm.
    It implements the property IFFT(X) = CONJ(FFT(CONJ(X))).
    The scaling factor implicitly aligns with the `fft_py` function.

    x_in (np.ndarray): Input FFT result (complex numbers).
    Returns (np.ndarray): The IFFT of the input.
    """
    n = len(x_in)
    # 'x_out' will be the final output buffer, initially empty.
    x_out = np.zeros(n, dtype=np.complex128)
    # 'y_in' will be the initial input buffer for fft0.
    # First, conjugate the input as per the C++ `ifft` logic.
    y_in = np.conj(np.array(x_in, dtype=np.complex128))

    # Call fft0_py_corrected on the conjugated data.
    # x_out is the designated output, y_in is the designated input (holding conjugated data).
    # 'eo' starts as False.
    fft0_py_corrected(n, 1, False, x_out, y_in)

    # Conjugate the result back as per the C++ `ifft` logic.
    # No additional 1/N scaling is needed here, as the scaling from `fft_py` is implicitly handled.
    return np.conj(x_out)

# Test functions
def test_fft_ifft_corrected():
    """
    Tests the custom FFT and IFFT implementations against NumPy's functions.
    """
    # Test with N = 8 (power of 2)
    N = 8
    # Create a random complex signal
    np.random.seed(42) # For reproducibility
    signal = np.random.rand(N) + 1j * np.random.rand(N)
    print("Original Signal:")
    print(signal)

    # --- Test Forward FFT ---
    # Our FFT function (fft_py)
    fft_result_my = fft_py(signal) # Pass the signal array

    # NumPy's FFT (Note: NumPy's fft does not scale by 1/N by default)
    numpy_fft_result = np.fft.fft(signal)

    print("\nMy FFT Result (scaled by 1/N as per C++):")
    print(fft_result_my)
    print("\nNumPy FFT Result (unscaled):")
    print(numpy_fft_result)

    # Comparison: My FFT result should be NumPy's FFT result divided by N.
    assert np.allclose(fft_result_my, numpy_fft_result / N, atol=1e-9), \
        "FFT results do not match NumPy's FFT (after accounting for 1/N scaling)!"
    print("\nFFT Test Passed: My FFT results match NumPy's (scaled by 1/N).")

    # --- Test Inverse FFT ---
    # Perform IFFT using my function, using the result from my FFT.
    # According to the C++ code's scaling and properties:
    # my_fft(signal) -> X_hat = DFT_true(signal) / N
    # my_ifft(X_hat) -> Should recover the original signal.
    ifft_result_my = ifft_py(fft_result_my)

    print("\nMy IFFT Result (from my FFT result):")
    print(ifft_result_my)

    # Check if IFFT recovers the original signal
    assert np.allclose(ifft_result_my, signal, atol=1e-9), \
        "IFFT did not recover the original signal!"
    print("\nIFFT Test Passed: My IFFT correctly recovers the original signal.")

    # --- Additional Test: My IFFT with NumPy's FFT result ---
    # If we feed NumPy's unscaled FFT result (DFT_true(signal)) into my IFFT,
    # my IFFT calculates conj(DFT_true(conj(DFT_true(signal)))).
    # Since DFT_true(conj(X)) = conj(IDFT_true(X)) * N,
    # this becomes conj(conj(IDFT_true(DFT_true(signal))) * N)
    # = IDFT_true(DFT_true(signal)) * N = signal * N.
    ifft_result_my_from_numpy = ifft_py(numpy_fft_result)
    print("\nMy IFFT Result (from NumPy FFT result, expecting original_signal * N):")
    print(ifft_result_my_from_numpy)
    assert np.allclose(ifft_result_my_from_numpy, signal * N, atol=1e-9), \
        "My IFFT from NumPy FFT result does not match expected (signal * N)!"
    print("\nIFFT from NumPy FFT test passed (expected signal * N).")

    # --- Final check: NumPy's IFFT consistency ---
    numpy_ifft_result = np.fft.ifft(numpy_fft_result)
    print("\nNumPy IFFT Result (from NumPy FFT result):")
    print(numpy_ifft_result)
    assert np.allclose(numpy_ifft_result, signal, atol=1e-9), \
        "NumPy IFFT failed to recover original signal (internal check)!"
    print("\nNumPy IFFT internal consistency check passed.")

# Run the tests
if __name__ == "__main__":
    test_fft_ifft_corrected()