import numpy as np
import math 

def stockham(n, s, eo, x, y):
    m = n // 2
    theta0 = math.pi / s

    if (n == 0): return
    if (n == 1 and eo):
        for q in range(s):
            y[q] = x[q]
    else:
        for p in range(int(m)):
            for q in range(s):
                wq = math.cos(q * theta0) - 1j * math.sin(q * theta0)
                a = x[q + s*(p + 0)]
                b = x[q + s*(p + m)] * wq;
                y[q + s*(2*p + 0)] = a + b;
                y[q + s*(2*p + 1)] = a - b;
               
        stockham(n//2, s * 2, not eo, y, x)

def fft(n, input):
    y = np.zeros(n, dtype=np.complex64)
    stockham(n, 1, False, input, y)
    return y  # Scale the output by 1/N as per FFT definition



def test_fft_with_random_data(sizes):
    """
    Tests an FFT function against numpy.fft.fft with random complex data
    for specified input sizes (preferably powers of 2 for typical FFT algorithms).

    Args:
        fft_func (callable): The user's FFT function to test.
        sizes (list): A list of integer sizes to test (e.g., [8, 16, 32]).
    """
    print("Starting FFT function tests with random data...\n")

    for N in sizes:
        print(f"--- Testing for N = {N} ---")
        # Generate random complex data (real and imaginary parts between -1 and 1)
        # Ensure data type is complex64 as specified in your prompt example.
        random_input = (np.random.uniform(-1, 1, N) + 1j * np.random.uniform(-1, 1, N)).astype(np.complex64)

        print(f"Input data (first 5 elements): {random_input[:min(5, N)]}{'...' if N > 5 else ''}")

        # Compute FFT using numpy's implementation
        numpy_fft_output = np.fft.fft(random_input)
        print(f"Numpy FFT output (first 5 elements): {numpy_fft_output[:min(5, N)]}{'...' if N > 5 else ''}")

        # Compute FFT using the provided (your) function
        custom_fft_output = fft(N, random_input) # Pass the original complex64 input
        print(f"Custom FFT output (first 5 elements): {custom_fft_output[:min(5, N)]}{'...' if N > 5 else ''}")

        # Compare the results using np.allclose for floating-point accuracy.
        # rtol (relative tolerance) and atol (absolute tolerance) are typical values.
        are_close = np.allclose(numpy_fft_output, custom_fft_output, rtol=1e-3, atol=1e-4)
        print(f"Results are approximately equal (numpy vs custom): {are_close}")

        if not are_close:
            diff = np.abs(numpy_fft_output - custom_fft_output)
            max_diff = np.max(diff)
            print(f"  WARNING: Mismatch detected! Max absolute difference: {max_diff:.6e}")
            # Uncomment the line below for more detailed debugging of differences:
            # print(f"  Detailed differences (first 5): {diff[:min(5, N)]}")
        print("-" * (20 + len(str(N))) + "\n")

    print("FFT function tests completed.")


if __name__ == "__main__":
    # Define the sizes for testing. "2 poers" is interpreted as two powers of 2.
    # You can modify this list to test different lengths,
    # especially other powers of 2 like [32, 64, 128].
    test_sizes = [2**i for i in range(25)] # Common power-of-2 sizes for FFT testing

    # Run the tests with random data for the specified sizes
    test_fft_with_random_data(test_sizes)
