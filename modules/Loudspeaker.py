"""
Loudspeaker.py

This module defines the Loudspeaker class which is used for analyzing and modeling the behavior of loudspeakers. 
It includes functionality for calculating electrical and mechanical impedance, estimating various parameters, 
and plotting relevant data for analysis.

Classes:
    Loudspeaker: Represents a loudspeaker and provides methods for analyzing its properties.

Functions:
    freq_limits: Helper function to limit frequencies for analysis.
    find_resonance_frequency: Function to find the resonance frequency from the given data.
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.ElectricalImpedance import ElectricalImpedance
from modules.MechanicalImpedance import MechanicalImpedance


def freq_limits(f_axis, f_min, f_max, *variables):
    """
    Limit the frequency range of the provided data.

    Args:
        f_axis (array): Array of frequency values.
        f_min (float): Minimum frequency limit.
        f_max (float): Maximum frequency limit.
        variables (list): List of variables to limit.

    Returns:
        list: List of variables limited to the specified frequency range.
    """
    f1_index = np.abs(f_axis - f_min).argmin()
    f2_index = np.abs(f_axis - f_max).argmin()

    new_variables = []
    for variable in variables:
        new_variables.append(variable[f1_index:f2_index])

    return new_variables


def find_resonance_frequency(f, VI):
    """
    Find the resonance frequency from voltage and current data.

    Args:
        f (array): Frequency array.
        VI (array): Voltage/Current data.

    Returns:
        float: Resonance frequency.
    """
    f_index = np.argmax(np.abs(VI))
    return f[f_index]


class Loudspeaker:
    """
    Represents a loudspeaker for analyzing its electrical and mechanical properties.

    The class provides methods to estimate resonance frequency, electrical and mechanical impedances,
    and to plot these properties for analysis.

    Methods:
        __init__: Constructor to initialize the loudspeaker with given data.
        Bl_estimate: Estimates the Bl product.
        Ze_from_measurement: Calculates electrical impedance from measurements.
        Zm_from_measurement: Calculates mechanical impedance from measurements.
        plot_Ze: Plots electrical impedance.
        plot_Zm: Plots mechanical impedance.
        ... [other methods] ...
    """

    def __init__(self, f_axis, U, I, V, model_Ze='RL', model_Zm='mass-spring', f_min=20, f_max=10e3, displacement_delay_sec=650e-6):
        """
        Initialize the Loudspeaker object with measurement data and models.

        Args:
            f_axis (array): Frequency axis for the analysis.
            U (array): Voltage measurement data.
            I (array): Current measurement data.
            V (array): Velocity measurement data.
            model_Ze (str): Model type for electrical impedance (default 'RL').
            model_Zm (str): Model type for mechanical impedance (default 'mass-spring').
            f_min (float): Minimum frequency for analysis (default 20 Hz).
            f_max (float): Maximum frequency for analysis (default 10 kHz).
            displacement_delay_sec (float): Displacement delay in seconds (default 650 µs).
        """

        # limit frequencies
        U, I, V, f_axis = freq_limits(f_axis, f_min, f_max, U, I, V, f_axis)

        # frequency limits and axes
        self.f_min = f_min
        self.f_max = f_max
        self.f_axis = f_axis
        self.omega = 2*np.pi*f_axis

        # Displacement Delay Compensate
        V *= np.exp(1j*self.omega*displacement_delay_sec)

        # save variables to the object
        self.U = U
        self.I = I
        self.V = V

        # Resonance Frequency Estimation
        self.f_res = find_resonance_frequency(f_axis, np.abs(V/I))

        # Bl estimation
        self.Bl = self.Bl_estimate()

        # save measured data of electrical and mechanical impedance
        self.Ze_measured_data = Loudspeaker.Ze_from_measurement(
            U, I, V, self.Bl)
        self.Zm_measured_data = Loudspeaker.Zm_from_measurement(I, V, self.Bl)

        # estimation of paramet
        self.Ze_model_object = ElectricalImpedance(
            self.omega, self.Ze_measured_data, model_Ze)

        # Kms, Mms estimation
        Zm, w = freq_limits(f_axis, 1/2*self.f_res, 2*self.f_res,
                            self.Zm_measured_data, self.omega)

        self.Zm_model_object = MechanicalImpedance(w, Zm, model_Zm)

    def Bl_estimate(self, maxBl=50, stepBl=0.01):

        # limit the frequecies around the resonant frequency
        Ux, Ix, Vx = freq_limits(
            self.f_axis, self.f_res/2, self.f_res*2, self.U, self.I, self.V)

        # The loop to estimate the Bl value by minimizing the standard deviation
        # of the real part of the measured electrical impedance (Ze).
        # The loop iterates over a range of Bl_test values from -maxBl to maxBl
        # in steps of stepBl. These values are potential estimates for the Bl parameter.
        # Note that a negative value of Bl can happen due to inverted polarity of the
        # loudspeaker under test.
        Bl = 0
        std_test = np.Inf
        for Bl_test in np.arange(-maxBl, maxBl, stepBl):
            Ze = Loudspeaker.Ze_from_measurement(Ux, Ix, Vx, Bl_test)
            new_std = np.std(np.real(Ze))
            if new_std < std_test:
                std_test = new_std
                Bl = Bl_test

        return Bl

    @staticmethod
    def Ze_from_measurement(U, I, V, Bl):
        # Basic equation for electrical impedance
        return U/I - Bl*V/I

    @staticmethod
    def Zm_from_measurement(I, V, Bl):
        # Basic equation for mechanical impedance
        return Bl*I/V

    def print_parameters(self):
        print('-----------------------------------------')
        print(f"{'Bl (Force factor)':<30}: {np.abs(self.Bl):.2e} Tm")
        self.Ze_model_object.print_parameters()
        self.Zm_model_object.print_parameters()
        print(f"{'Fs (res. freq. from impedance)':<30}: {self.f_res:.2e} Hz")
        print('-----------------------------------------')

        # Inform the user if the polarity was inverted
        if self.Bl < 0:
            print(
                f"!!! Bl was estimated as Bl = {self.Bl:.2e} Tm (negative value) that indicates inverted polarity")
            print("of the loudspeaker terminals during the measurement.")

    def input_impedance_from_model(self):

        # -- Reconstruct the impedance curve and compare it with the measured one
        Ze = self.Ze_model_object.calculate_impedance(1j*self.omega)
        Zm = self.Zm_model_object.calculate_impedance(1j*self.omega)

        return Ze + self.Bl**2 / Zm

    def plot_input_impedance(self):

        # measued data
        Z_measured_data = self.U/self.I

        # model data
        Z_model_data = self.input_impedance_from_model()

        # find the maximum value to avoid matplotlib wrong auto y-limits
        maxZ = np.max([np.nanmax(np.abs(Z_model_data)),
                      np.nanmax(np.abs(Z_measured_data))])

        fig, ax = plt.subplots(2)
        ax[0].semilogx(self.f_axis, np.abs(
            Z_measured_data), label='measurement')
        ax[0].semilogx(self.f_axis, np.abs(Z_model_data), label='model')
        ax[0].set(xlim=[self.f_min, self.f_max], ylim=[0, 1.1*maxZ])
        ax[0].set(xlabel='Frequency [Hz]', ylabel='Absolute value [Ohm]')
        ax[0].legend()
        ax[0].grid()
        ax[1].semilogx(self.f_axis, np.angle(Z_measured_data))
        ax[1].semilogx(self.f_axis, np.angle(Z_model_data))
        ax[1].set(xlim=[self.f_min, self.f_max])
        ax[1].set(xlabel='Frequency [Hz]', ylabel='Phase [rad]')
        ax[1].grid()

        ax[0].set(title='Input Impedance')
        fig.tight_layout()
        plt.show()

    def plot_Ze(self):

        # measued data
        Ze_measured_data = Loudspeaker.Ze_from_measurement(
            self.U, self.I, self.V, self.Bl)

        # model data
        Ze_model_data = self.Ze_model_object.calculate_impedance(1j*self.omega)

        fig, ax = plt.subplots(2)
        ax[0].semilogx(self.f_axis, np.real(
            Ze_measured_data), label='measurement')
        ax[0].semilogx(self.f_axis, np.real(Ze_model_data), label='model')
        ax[0].set(xlim=[self.f_min, self.f_max])
        ax[0].set(xlabel='Frequency [Hz]', ylabel='Apparent Resistance [Ohm]')
        ax[0].legend()
        ax[0].grid()

        ax[1].semilogx(self.f_axis, 1000 *
                       np.imag(self.Ze_measured_data)/self.omega, label='measurement')
        ax[1].semilogx(self.f_axis, 1000 *
                       np.imag(Ze_model_data)/self.omega, label='model')
        ax[1].set(xlim=[self.f_min, self.f_max])
        ax[1].set(xlabel='Frequency [Hz]', ylabel='Apparent Inductance [mH]')
        ax[1].legend()
        ax[1].grid()

        ax[0].set(title='Electrical Impedance')
        fig.tight_layout()
        plt.show()

    def plot_Zm(self):

        # measued data
        Zm_measured_data = Loudspeaker.Zm_from_measurement(
            self.I, self.V, self.Bl)

        # model data
        Zm_model_data = self.Zm_model_object.calculate_impedance(1j*self.omega)

        fig, ax = plt.subplots(2)
        ax[0].loglog(self.f_axis, np.real(
            Zm_measured_data), label='measurement')
        ax[0].loglog(self.f_axis, np.real(Zm_model_data), label='model')
        ax[0].set(xlim=[20, 500], ylim=[.1, 10])
        ax[0].set(xlabel='Frequency [Hz]',
                  ylabel='Mechancial Resistance [Ohm]')
        ax[0].legend()
        ax[0].grid()

        ax[1].semilogx(self.f_axis, np.abs(
            1/Zm_measured_data), label='measurement')
        ax[1].semilogx(self.f_axis, np.abs(1/Zm_model_data), label='model')
        ax[1].set(xlim=[20, 500])
        ax[1].set(xlabel='Frequency [Hz]',
                  ylabel='Mechanical Admittance [m/Ns]')
        ax[1].legend()
        ax[1].grid()

        ax[0].set(title='Mechanical Impedance')
        fig.tight_layout()
        plt.show()
