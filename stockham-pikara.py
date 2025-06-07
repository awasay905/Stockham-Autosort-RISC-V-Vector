import numpy as np
import math 

def stockham(n, s, eo, x, y):
    m = n // 2
    theta0 = math.pi / s

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


input = [1,2,3,4,5,6,7,8]
print(np.fft.fft(input))  # For comparison with numpy's FFT
print(fft(len(input), np.array(input, dtype=np.complex64)))