import numpy as np
from scipy.sparse import hstack, vstack, coo_matrix
from math import *

class CQT :
    def __init__(self, fmin, fmax, bins, fs, wnd) :
        print 'Initializing...'
        self.eps = 1e-5
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        self.wnd  = wnd
        self.fs = fs
        self.Q = 1 / (pow(2, 1.0 / bins) - 1)
        K = int(ceil(bins * log(fmax / fmin) / log(2)))
        # print 'K:', K
        self.fftlen = int(pow(2, ceil(log(self.Q * fs / fmin) / log(2))))
        self.ker = []
        for k in range(K, 0, -1) :
            N = ceil(self.Q * fs / (fmin * pow(2, (k - 1.0) / bins)))
            tmpKer = wnd(N) * np.exp(2 * pi * 1j * self.Q * np.arange(N) / N) / N;
            ker = np.fft.fft(tmpKer, self.fftlen)
            # ker = np.select([abs(ker) > self.eps], [ker])
            self.ker += [coo_matrix(ker, dtype = np.complex128)]
        # print 'shape:', hstack(self.ker).tocsc().shape
        self.ker.reverse()
        self.ker = vstack(self.ker).tocsc().transpose().conj() / self.fftlen
        print 'Initialized OK.'

    def fast(self, x) :
        # print self.ker.shape, np.fft.fft(x, self.fftlen).shape
        return (np.fft.fft(x, self.fftlen).reshape(1, self.fftlen) * self.ker)[0]

    def slow(self, x) :
        cq = []
        for k in range(1, int(ceil(self.bins * log(self.fmax / self.fmin) / log(2))) + 1) :
            N = ceil(self.Q * self.fs / (self.fmin * pow(2, (k - 1.0) / self.bins)))
            # print x[:N].shape, (wnd(N) * np.exp(2 * pi * 1j * np.arange(N) / N) / N).shape
            cq += [x[:N].dot(self.wnd(N) * np.exp(-2 * pi * 1j * self.Q * np.arange(N) / N) / N)]
        return np.array(cq)

def hamming(length) :
    return 0.46 - 0.54 * np.cos(2 * pi * np.arange(length) / length)

# np.arange()
def test() :
    length = 44100
    x = np.random.random(length)
    drd = CQT(40, 22050, 12, 44100, hamming)
    
    y, z = drd.fast(x), drd.slow(x)
    print 'Benchmark the `EFFICIENT` method:'
    timeit('y = drd.fast(x)', repeat = 10)

    print 'Benchmark the `BRUTE-FORCE` method:'
    timeit('z = drd.slow(x)', repeat = 10)

    print 'The difference: ', max(abs(z - y))

test()
