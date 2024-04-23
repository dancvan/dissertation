############################
#### CALIBRATION script ####
############################

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ppt2latex2')
plt.rcParams["font.family"] = "Times New Roman"

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
ef_eff      = 13.3                  # [[V/m]/V]
#ef_eff      = 42                    # [[V/m]/V]
v2hz        = 1.7e6                 # [Hz/V]
c           = 299792458             # [m/s]
L_cav       = .105                  # [m]
nu_laser    = c/(1064e-9)           # [m]

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
Gf_mag_data = np.loadtxt(G_dir + 'SCRN0530.TXT').transpose()
Gf_phase_data = np.loadtxt(G_dir + 'SCRN0531.TXT').transpose()
Gs_mag_data = np.loadtxt(G_dir + 'SCRN0534.TXT').transpose()
Gs_phase_data = np.loadtxt(G_dir + 'SCRN0535.TXT').transpose()

Gf = transfer_function(Gf_mag_data[1], Gf_phase_data[1])
Gs = transfer_function(Gs_mag_data[1], Gs_phase_data[1])

## A1 (HVA CH3) -> in-loop
A1_dir = home_dir + 'HVA_3ch/'
A1_mag_data = np.loadtxt(A1_dir + 'SCRN0494.TXT').transpose()
A1_phase_data = np.loadtxt(A1_dir + 'SCRN0495.TXT').transpose()
A1 = transfer_function(A1_mag_data[1], A1_phase_data[1])

## SR560 (Summing port)
SR560_dir = home_dir + 'SR560/'
SR560_mag_data = np.loadtxt(SR560_dir + 'SCRN0454.TXT').transpose()
SR560_phase_data = np.loadtxt(SR560_dir + 'SCRN0455.TXT').transpose()
SR560_mag2_data = np.loadtxt(SR560_dir + 'SCRN0456.TXT').transpose()
SR560_phase2_data = np.loadtxt(SR560_dir + 'SCRN0457.TXT').transpose()
SR560_tf1 = transfer_function(SR560_mag_data[1], SR560_phase_data[1])
SR560_tf2 = transfer_function(SR560_mag2_data[1], SR560_phase2_data[1])

## A2 (HVA trek) -> electrodes
A2_dir = home_dir + 'HVA_trek/'
A2_mag_data = np.loadtxt(A2_dir + 'trek_mag.TXT').transpose()
A2_phase_data = np.loadtxt(A2_dir + 'trek_phase.TXT').transpose()
A2 = transfer_function(A2_mag_data[1], A2_phase_data[1])

# Compute coupling efficiency (C)
C = lambda tf, G_, E : (L_cav/nu_laser)*(tf*(1-G_)*A1*(v2hz*SR560_tf1))/(G_*A2*E)

C_fast = C(tf_fast, Gf, ef_eff)
C_slow = C(tf_slow, Gs, ef_eff)
diff = C_slow - C_fast

## Compute differential (between fast and slow axes)
C_slow_mag = np.sqrt(C_slow*np.conj(C_slow))
C_fast_mag = np.sqrt(C_fast*np.conj(C_fast))
diff_mag = np.sqrt(diff*np.conj(diff))

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.loglog(tffastmag_data[0], C_slow_mag, label = 'slow\_axis')
ax1.loglog(tffastmag_data[0], C_fast_mag, label = 'fast\_axis')
ax1.loglog(tffastmag_data[0], diff_mag, alpha=.5, label='differential')
ax1.hlines(y=7e-17, xmin=20e3, xmax=40e3, linestyle='--', linewidth=5.0, color='m')
ax1.tick_params(axis='y', which='minor')
ax1.set_xlim(tffastmag_data[0][0], tffastmag_data[0][-1])
ax1.set_ylabel('Coupling [[m]/[V/m]]')
ax1.legend()
ax2.semilogx(tffastmag_data[0], (np.arctan2(np.imag(C_slow), np.real(C_slow)))*(180/np.pi), label = 'slow\_axis')
ax2.semilogx(tffastmag_data[0], (np.arctan2(np.imag(C_fast), np.real(C_fast)))*(180/np.pi), label = 'fast\_axis')
ax2.semilogx(tffastmag_data[0], (np.arctan2(np.imag(diff), np.real(diff)))*(180/np.pi) , alpha=.5, label = 'differential')
ax2.set_xlim(tffastmag_data[0][0], tffastmag_data[0][-1])
ax2.set_ylim(-180.0, 180.0)
ax2.legend()
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [deg]')
#plt.show()
plt.savefig('../../../figs/ALGAAS/coupling_tf.pdf', dpi=300, format='pdf', bbox_inches='tight')


