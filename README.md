
# Loudspeaker Measurement Project

## Overview
This project is designed for the measurement of loudspeaker parameters measuring the voltage and current of the loudspeaker, and using a displacement sensor to get the membrane velocity.

## Setup and Installation
The project may require external libraries like `numpy`, `scipy`, `matplotlib`, and `uldaq` (for the DT9837 device).

## Running the Project
To run the main script (`speaker_measurement.py`), execute it from the command line.

```bash
python speaker_measurement.py
```

## Modules Description
1. **ElectricalImpedance.py**: Models and estimates the electrical impedance of loudspeakers.
2. **instru_temp.py**: Measures environmental parameters such as temperature and humidity.
3. **Loudspeaker.py**: Integrates electrical and mechanical impedance models for loudspeaker analysis.
4. **measurement_DT9837.py**: Handles data acquisition using a DT9837 device.
5. **MechanicalImpedance.py**: Models and estimates the mechanical impedance of loudspeakers.
6. **SynchSweptSine.py**: Generates synchronized swept sine signal for measurement of the loudspeaker's response.

## Tutorial
For detailed instructions on setting up the displacement sensor, refer to the following video tutorial:
[![IMAGE ALT TEXT HERE](https://ant-novak.com/_others/laum/tuto_capteur_deplacement.jpg)](https://ant-novak.com/_others/laum/tuto_capteur_deplacement.mp4)

## Data
The `data` folder contains sample measurement data. The measurement data are saved in this folder.

## Dependencies
Install all the required libraries.

```bash
pip install numpy scipy matplotlib
```


