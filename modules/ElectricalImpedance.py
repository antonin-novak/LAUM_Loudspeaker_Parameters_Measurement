import numpy as np
from scipy.optimize import least_squares


class ElectricalImpedance:
    ''' 
    Model of loudspeaker Electrical Impedance for parameter estimation.

    This class models the electrical impedance of a loudspeaker and provides
    methods to estimate parameters for specific models such as Leach, R2L2, and R3L3.
    Each model represents a different approach to modeling loudspeaker impedance,
    taking into account (or not) eddy currents.

    Usage:
    -----
    # Example usage demonstrating how to create an instance, estimate parameters, and plot impedance.
    omega = 2*np.pi*f_axis  # angular frequency
    my_model = ElectricalImpedance(omega, Ze_measured_data, Re_estimated=Re, model='Leach')
    my_model.print_parameters()
    Ze_model_data = my_model.calculate_impedance(omega)
    # Plotting code...

    Attributes:
    ----------
    Re : float
        Estimated resistance (Re) from previous measurements.
    params: list
        List of estimated parameters for the selected model.

    Methods:
    -------    
    costFunction(self, parameters, omega, Ze_measured_data):
        Calculates the error between the modeled and measured impedance.
        Used internally by least squares optimization.

    Model-specific Methods (RL, Leach, R2L2, R3L3):
        Each method corresponds to a specific impedance model and calculates impedance based on its formula.

        RL_model
            basic Thiele-Smalle R-L model (resistance impedance)
            Ze = Re + Le*jw
        Leach_model
            Leach model
            Ze = Re + K*jw**beta
        R2L2_model
            R2L2 model
            Ze = Re + Le*jw + jw*R2*L2/(R2 + jw*L2)
        R3L3_model
            R3L3 model
            Ze = Re + Le*jw + jw*R2*L2/(R2 + jw*L2) + jw*R3*L3/(R3 + jw*L3)

    Print Methods (RL_print, Leach_print, R2L2_print, R3L3_print):
        Print estimated parameters for each model, aiding in interpretation and validation.

    Author: Antonin Novak
    Version: 1.0
    Last Updated: 4.12.2023
    '''

    def __init__(self, omega, Ze_measured_data, model):

        self.Re = np.real(Ze_measured_data[0])

        # Model methods mapping
        self.model_methods = {
            'RL': (self.RL_model, self.RL_print, [1e-3], [1e-2]),
            'Leach': (self.Leach_model, self.Leach_print, [1e-3, 1], [1e-2, 10]),
            'R2L2': (self.R2L2_model, self.R2L2_print, [1e-3, 1e-3, 0.1], [1e-2, 1e-2, 10]),
            'R3L3': (self.R3L3_model, self.R3L3_print, [1e-3, 1e-3, 0.1, 1e-3, 1], [1e-2, 1e-2, 10, 1e-2, 10])
        }

        # Set model-specific methods and parameters
        if model in self.model_methods:
            self.calculate_impedance = self.model_methods[model][0]
            self.print_parameters = self.model_methods[model][1]
            self.guess = self.model_methods[model][2]
            self.bounds = self.model_methods[model][3]
        else:
            raise ValueError(
                f"Model '{model}' not recognized. Available models: {list(self.model_methods.keys())}")

        self.params = least_squares(self.costFunction,
                                    self.guess,
                                    bounds=(0, self.bounds),
                                    args=(omega, Ze_measured_data)).x

    def costFunction(self, parameters, omega, Ze_measured_data):
        self.params = parameters
        Fit = self.calculate_impedance(1j*omega)
        return (np.abs(Ze_measured_data) - np.abs(Fit))

    def RL_model(self, s):
        Le = self.params[0]
        return self.Re + Le*s

    def RL_print(self):
        Le = self.params[0]
        print(f"{'Re (Resistance)':<30}: {self.Re:.2e} Ohm")
        print(f"{'Le (Inductance)':<30}: {Le:.2e} H")

    def Leach_model(self, s):
        K, beta = self.params
        return self.Re + K*s**beta

    def Leach_print(self):
        K, beta = self.params
        print(f"{'Re (Resistance)':<30}: {self.Re:.2e} Ohm")
        print(f"{'K (Constant)':<30}: {K:.2e} H")
        print(f"{'beta (Exponent)':<30}: {beta:.2e}")

    def R2L2_model(self, s):
        Le, L2, R2 = self.params
        return self.Re + Le*s + s*R2*L2/(R2 + s*L2)

    def R2L2_print(self):
        Le, L2, R2 = self.params
        print(f"{'Re (Resistance)':<30}: {self.Re:.2e} Ohm")
        print(f"{'Le (Inductance)':<30}: {Le:.2e} H")
        print(f"{'R2 (Resistance)':<30}: {R2:.2e} Ohm")
        print(f"{'L2 (Inductance)':<30}: {L2:.2e} H")

    def R3L3_model(self, s):
        Le, L2, R2, L3, R3 = self.params
        return self.Re + Le*s + s*R2*L2/(R2 + s*L2) + s*R3*L3/(R3 + s*L3)

    def R3L3_print(self):
        Le, L2, R2, L3, R3 = self.params
        print(f"{'Re (Resistance)':<30}: {self.Re:.2e} Ohm")
        print(f"{'Le (Inductance)':<30}: {Le:.2e} H")
        print(f"{'R2 (Resistance)':<30}: {R2:.2e} Ohm")
        print(f"{'L2 (Inductance)':<30}: {L2:.2e} H")
        print(f"{'R3 (Resistance)':<30}: {R3:.2e} Ohm")
        print(f"{'L3 (Inductance)':<30}: {L3:.2e} H")
