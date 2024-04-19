############################
#### CALIBRATION script ####
############################

import numpy as np
import matplotlib.pyplot as plt

def transfer_function(amplitude, phase,load_corr=False,anal_tag='sr785',\
zload=50):
    """
    Takes frequency response data (amplitude and phase) combines it into a
    complex vector
    """
    volt_div = lambda a, b : a/(a + b)

    if load_corr == True:
        if anal_tag == 'sr785':
            corr = volt_div(1e6,zload)
        if anal_tag == 'agilent':
            corr = volt_div(50,zload)
    else:
        corr = 1

    return 10**(amplitude/20)* np.exp(1j*(phase/180)*np.pi)/corr

# Constants
ef_eff  = 42                 # [[V/m]/V]
v2hz    = 1.7e6            # [Hz / V]

# Relevant data imports

## tf imports

tffastmag_data = np.loadtxt('fast/' + 'db.TXT').transpose()
tffastphase_data = np.loadtxt('fast/' + 'deg.TXT').transpose()
tf_fast = transfer_function(tffastmag_data[1], tffastphase_data[1])

tfslowmag_data = np.loadtxt('slow/' + 'db.TXT').transpose()
tfslowphase_data = np.loadtxt('slow/' + 'deg.TXT').transpose()
tf_slow = transfer_function(tfslowmag_data[1], tfslowphase_data[1])

home_dir = 'calib_tfs/'

## OLG
G_dir = home_dir + 'OLG/'
G_mag_data = np.loadtxt(G_dir + 'SCRN0530.TXT').transpose()
G_phase_data = np.loadtxt(G_dir + 'SCRN0531.TXT').transpose()
G = transfer_function(G_mag_data[1], G_phase_data[1])

## A1 (HVA CH3) -> in-loop
A1_dir = home_dir + 'HVA_3ch/'
A1_mag_data = np.loadtxt(A1_dir + 'SCRN0494.TXT').transpose()
A1_phase_data = np.loadtxt(A1_dir + 'SCRN0495.TXT').transpose()
A1 = transfer_function(A1_mag_data[1], A1_phase_data[1])

A1
## A2 (HVA trek) -> electrodes
A2_dir = home_dir + 'HVA_trek/'
A2_mag_data = np.loadtxt(A2_dir + 'trek_mag.TXT').transpose()
A2_phase_data = np.loadtxt(A2_dir + 'trek_phase.TXT').transpose()
A2 = transfer_function(A2_mag_data[1], A2_phase_data[1])

# Compute coupling efficiency (C)
C = lambda tf, E : tf*(1+G)*A1/(G*A2*E)


C(tf_fast, ef_eff)
