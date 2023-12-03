import warnings
import os

import numpy as np
from modules.SynchSweptSine import SynchSweptSine
from modules.measurement_DT9837 import measurement_DT9837
from modules.instru_temp import get_temperature_humidity
from modules.Loudspeaker import Loudspeaker

os.chdir(os.path.dirname(os.path.realpath(__file__)))
warnings.filterwarnings('ignore')

''' ------------------------------------------- '''

fs = 48000
voltage_sensitivity = 1.014  # V/V
current_sensitivity = 1.004   # V/A
displacement_sensitivity = 1.25/5e-3  # V/mm
velocity_sensitivity = 1/100e-3  # V/m/s

description = {'Project name': "Test",
               'description': 'test',
               'voltage-sensitivity': voltage_sensitivity,
               'current-sensitivity': current_sensitivity,
               'velocity-sensitivity': velocity_sensitivity,
               'temperature-humidity': get_temperature_humidity()}


''' --------- Signal Definition --------- '''
# swept-sine signal
s = SynchSweptSine(f1=10, f2=20e3, fs=fs, T=5, fade=[int(fs/10), int(fs/100)])
x = np.concatenate((np.zeros(int(fs/10)), s.signal()))

''' --------- Measurement --------- '''
MEAS = False

filename = "test_meas"

if MEAS:
    in_buffer = measurement_DT9837(0.8*x, fs)
    u = 1/voltage_sensitivity * np.array(in_buffer[0::4])
    i = 1/current_sensitivity * np.array(in_buffer[1::4])
    x = 1/displacement_sensitivity * np.array(in_buffer[3::4])
    v = 1/velocity_sensitivity * np.array(in_buffer[2::4])

    ''' --------- Data Extraction --------- '''
    hU, hI, hX, hV = s.getIR(u), s.getIR(i), s.getIR(x), s.getIR(v)

    data = {'Speaker Name': 'Test'}
    data['description'] = description
    data['hU'], data['hI'], data['hX'], data['hV'] = hU, hI, hX, hV

    writeFile = 'y'
    if os.path.exists(filename):
        print('---------------------')
        print('File exists. Do you want to overwite it (y/n)?')
        writeFile = input()
    if writeFile == 'y':
        np.save(f'./data/{filename}', data)
        print('File SAVED !!!')

else:
    data = np.load(f'./data/{filename}.npy', allow_pickle=True).item()
    hU, hI, hX, hV = data['hU'], data['hI'], data['hX'], data['hV']


''' Frequency domain '''
U = np.fft.rfft(hU[0:fs])
I = np.fft.rfft(hI[0:fs])
V = np.fft.rfft(hV[0:fs])
f_axis = np.fft.rfftfreq(fs, 1/fs)


# list of Ze models: RL, Leach, R2L2, R3L3
# list of Zm models: mass-spring, exp

speaker = Loudspeaker(f_axis, U, I, V, model_Ze='Leach',
                      model_Zm='mass-spring')
speaker.print_parameters()
speaker.plot_input_impedance()

speaker.plot_Ze()
speaker.plot_Zm()
