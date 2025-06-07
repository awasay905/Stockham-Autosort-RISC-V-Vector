#include <complex>
#include <cmath>

typedef std::complex<double> complex_t;

void fft0(int n, int s, bool eo, complex_t* x, complex_t* y)
{
    const int m = n/2;
    const double theta0 = M_PI/s;

    if (n == 1) { if (eo) for (int q = 0; q < s; q++) y[q] = x[q]; }
    else {
        for (int p = 0; p < m; p++) {
            for (int q = 0; q < s; q++) {
                const complex_t wq = complex_t(cos(q*theta0), -sin(q*theta0));
                const complex_t a = x[q + s*(p + 0)];
                const complex_t b = x[q + s*(p + m)] * wq;
                y[q + s*(2*p + 0)] = a + b;
                y[q + s*(2*p + 1)] = a - b;
            }
        }
        fft0(n/2, 2*s, !eo, y, x);
    }
}

void fft(int n, complex_t* x)
{
    complex_t* y = new complex_t[n];
    fft0(n, 1, 0, x, y);
    delete[] y;
    for (int k = 0; k < n; k++) x[k] /= n;
}

void ifft(int n, complex_t* x)
{
    for (int p = 0; p < n; p++) x[p] = conj(x[p]);
    complex_t* y = new complex_t[n];
    fft0(n, 1, 0, x, y);
    delete[] y;
    for (int k = 0; k < n; k++) x[k] = conj(x[k]);
}

