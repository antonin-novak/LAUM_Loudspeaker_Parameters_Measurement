import numpy as np
from scipy.optimize import least_squares


class MechanicalImpedance:
    ''' 
    Model of loudspeaker Mechanical Impedance for parameter estimation.

    This class models the mechanical impedance of a loudspeaker and provides
    methods to estimate parameters for specific models such as mass-spring, exponential, ...

    Usage:
    -----
    # Example usage demonstrating how to create an instance, estimate parameters, and plot impedance.
    omega = 2*np.pi*f_axis  # angular frequency
    my_model = MechanicalImpedance(omega, Zm_measured_data, model='mass-spring')
    my_model.print_parameters()
    Zm_model_data = my_model.calculate_impedance(omega)
    # Plotting code...

    Attributes:
    ----------
    params: list
        List of estimated parameters for the selected model.

    Methods:
    -------
    costFunction(self, parameters, omega, Zm_measured_data):
        Calculates the error between the modeled and measured impedance.
        Used internally by least squares optimization.

    Model-specific Methods (mass_spring_model, exp_model):
        Each method corresponds to a specific mechanical impedance model and calculates impedance based on its formula.

        mass_spring_model
            basic 1-dof mass-spring model
            Zm = Mms*jw + Rms + Kms/jw

        exp_model
            Knudsen-Jensen exponential model
            Zm = Mms*jw + Rms + 1/(Ce*jw**(1-beta))

        log_model
            Knudsen-Jensen log model
            Zm = Mms*jw + Rms + 1/(s*Cl*(1-lamb*np.log(s)))

        novak_model
            Model taking into account elastic and viscous losses
            Zm = Mms*jw + Rv + eta*jw**(beta-1) + 1/(C0*jw)

    Print Methods (mass_spring_print, exp_print):
        Print estimated parameters for each model, aiding in interpretation and validation.

    Author: Antonin Novak
    Version: 1.0
    Last Updated: 1.12.2023
    '''

    def __init__(self, omega, Zm_measured_data, model):

        self.omega = omega

        # Model methods mapping
        self.model_methods = {
            'mass-spring': (self.mass_spring_model, self.mass_spring_print, [1e-2, 1, 1000], [1e-1, 100, 10000]),
            'exp': (self.exp_model, self.exp_print, [1e-2, 1, 1e-3, 0], [1e-1, 100, 1e-2, 0.5]),
            'log': (self.log_model, self.log_print, [1e-2, 1, 1e-3, 0], [1e-1, 100, 1e-2, 1]),
            # Mms, Rv, eta, beta, C0
            'novak': (self.novak_model, self.novak_print, [1e-2, 1, 100, 0, 1e-3], [1e-1, 100, 10000, 1, 1e-2]),
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
                                    args=(omega, Zm_measured_data)).x

    def costFunction(self, parameters, omega, Zm_measured_data):
        self.params = parameters
        Fit = self.calculate_impedance(1j*omega)
        return (np.abs(Zm_measured_data) - np.abs(Fit))

    def resonance_frequency(self):
        # find the resonance frequency from the imaginary part of the mechanical impedance (when it passes through zero)
        Zm = self.calculate_impedance(1j*self.omega)
        f_axis = self.omega/(2*np.pi)

        # Find the index where Zm crosses zero
        f_idx = np.where(np.diff(np.sign(np.imag(Zm))))[0][0]

        # Interpolate to find the approximate zero crossing frequency
        x1, x2 = f_axis[f_idx], f_axis[f_idx + 1]
        y1, y2 = np.imag(Zm[f_idx]), np.imag(Zm[f_idx + 1])
        f_res = x1 - y1 * (x2 - x1) / (y2 - y1)

        return f_res

    def print_resonant_frequency(self):
        Fs = self.resonance_frequency()
        print(f"{'Fs (res. freq. from model)':<30}: {Fs:.2e} Hz")

    def mass_spring_model(self, s):
        Mms, Rms, Kms = self.params
        return Mms*s + Rms + Kms/s

    def mass_spring_print(self):
        Mms, Rms, Kms = self.params
        Fs = 1/(2*np.pi)*np.sqrt(Kms/Mms)
        print(f"{'Mms (Mass)':<30}: {Mms:.2e} kg")
        print(f"{'Rms (Mech. Resistance)':<30}: {Rms:.2e} Ns/m")
        print(f"{'Kms (Stiffness)':<30}: {Kms:.2e} N/m")
        print(f"{'... Cms (Complience)':<30}: {1/Kms:.2e} m/N")
        print(f"{'Fs (res. freq. from model)':<30}: {Fs:.2e} Hz")

    def exp_model(self, s):
        Mms, Rms, Ce, beta = self.params
        return Mms*s + Rms + 1/(Ce*s**(1-beta))

    def exp_print(self):
        Mms, Rms, Ce, beta = self.params
        print(f"{'Mms (Mass)':<30}: {Mms:.2e} kg")
        print(f"{'Rms (Mech. Resistance)':<30}: {Rms:.2e} Ns/m")
        print(f"{'Ce (Complience parameter)':<30}: {Ce:.2e} m/N")
        print(f"{'beta (exp. parameter)':<30}: {beta:.2e}")
        self.print_resonant_frequency()

    def log_model(self, s):
        Mms, Rms, Cl, lamb = self.params
        return Mms*s + Rms + 1/(s*Cl*(1-lamb*np.log(s)))

    def log_print(self):
        Mms, Rms, Cl, lamb = self.params
        print(f"{'Mms (Mass)':<30}: {Mms:.2e} kg")
        print(f"{'Rms (Mech. Resistance)':<30}: {Rms:.2e} Ns/m")
        print(f"{'Cl (Complience parameter)':<30}: {Cl:.2e}")
        print(f"{'lambda (log. parameter)':<30}: {lamb:.2e}")
        self.print_resonant_frequency()

    def novak_model(self, s):
        Mms, Rv, eta, beta, C0 = self.params
        return Mms*s + Rv + eta*s**(beta-1) + 1/(C0*s)

    def novak_print(self):
        Mms, Rv, eta, beta, C0 = self.params
        print(f"{'Mms (Mass)':<30}: {Mms:.2e} kg")
        print(f"{'Rv (viscouss loss coefficient)':<30}: {Rv:.2e} Ns/m")
        print(f"{'eta (elastic loss coefficient)':<30}: {eta:.2e}")
        print(f"{'beta (elastic loss exponent coefficient)':<30}: {beta:.2e}")
        print(f"{'C0 (Complience parameter)':<30}: {C0:.2e} m/N")
        self.print_resonant_frequency()
