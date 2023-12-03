import numpy as np


class SynchSweptSine():
    def __init__(self, f1=20, f2=20e3, fs=48e3, T=5, fade=[9600, 960]):
        self.f1 = f1
        self.f2 = f2
        self.fs = fs
        self.T = T
        self.fade = fade
        self.L = T/np.log(f2/f1)

    def t_axis(self):
        return np.arange(0, np.round(self.fs*self.T-1)/self.fs, 1/self.fs)

    def signal(self):
        t = self.t_axis()
        s = np.sin(2*np.pi*self.f1*self.L*np.exp(t/self.L))

        fade = self.fade
        # fade-in the input signal
        if self.fade[0] > 0:
            s[0:fade[0]] = s[0:fade[0]] * \
                ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

        # fade-out the input signal
        if self.fade[1] > 0:
            s[-fade[1]:] = s[-fade[1]:] * \
                ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)

        return s

    def f_axis(self, Npts):
        return np.fft.rfftfreq(Npts, d=1.0/self.fs)

    def Xinv(self, Npts):
        # definition of the inferse filter in spectral domain
        # (Novak et al., "Synchronized swept-sine: Theory, application, and implementation."
        # Journal of the Audio Engineering Society 63.10 (2015): 786-798.
        # Eq.(43))
        f_axis = self.f_axis(Npts)
        Xinv = 2*np.sqrt(f_axis/self.L)*np.exp(-1j*2*np.pi *
                                               f_axis*self.L*(1-np.log(f_axis/self.f1)) + 1j*np.pi/4)
        Xinv[0] = 0j
        return Xinv

    def getFRF(self, y):
        # FFT of the output signal
        Y = np.fft.rfft(y)/self.fs

        # complete FRF
        H = Y*self.Xinv(len(y))

        return H

    def getIR(self, y):

        # complete FRF
        H = self.getFRF(y)

        # iFFT to get IR
        return np.fft.irfft(H)
