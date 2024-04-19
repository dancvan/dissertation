import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use('ppt2latexsubfig_smlegend')
rc('font', **{'family':'Times New Roman'})
rc('text', usetex=True)

## import time series
traces0 = pd.read_csv('TEK00045.CSV')
traces0

traces1 = np.loadtxt('TEK00045.CSV',skiprows=1, delimiter=',').transpose()
traces2 = np.loadtxt('TEK00046.CSV',skiprows=1, delimiter=',').transpose()
traces3 = np.loadtxt('scan_1kHz.CSV',skiprows=1, delimiter=',').transpose()

## pre-plot and generate curve labels for each time series
[plt.plot(traces1[0], traces1[i+1]) for i in range(0,5)]
[plt.plot(traces2[0], traces2[i+1]) for i in range(0,5)]
[plt.plot(traces3[0], traces3[i+1]) for i in range(0,5)]


## plot all data sets on separate vertical subplots (not shared x axis)
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
[ax1.plot(traces1[0], traces1[i+1]) for i in range(0,4)]
[ax2.plot(traces2[0], traces2[i+1]) for i in range(0,4)]
[ax3.plot(traces3[0], traces3[i+1]) for i in range(0,4)]


#t1_offset = traces1[0][np.where((traces1[1]==traces1[1].min()))[0][0]]
#t2_offset = traces2[0][np.where((traces2[1]==traces2[1].min()))[0][0]]

#traces1[0] = traces1[0] - t1_offset
#traces2[0] = traces2[0] - t2_offset

size1 = int(traces1[0].size/2)
size2 = int(traces2[0].size/2)

window_pt5 = 200

pmax1 = traces1[1][0:2000].sum()/traces1[1][0:2000].size
pmax2 = traces2[1][0:2000].sum()/traces2[1][0:2000].size

traces1[1] = traces1[1]/pmax1
traces2[1] = traces2[1]/pmax2


plt.plot(traces1[0], traces1[3])
plt.plot(traces2[0], traces2[3])


pzt_coeff = 1.7e6 # Hz / V

#v2t1 = ((.24-.005) - (.20+.0075))*1e3/(.0005*2)

v2t1 = ((.20 + .01/4) - ( .18 + .005*(8/9)))*1e3/(.001) 

#plt.figure(figsize=(30,19.5)) 
plt.plot(traces2[0][(size2-window_pt5):(size2+window_pt5)]*v2t1*pzt_coeff, traces2[1][(size2-window_pt5):(size2+window_pt5)], label='fast / slow axis ')
plt.plot(traces1[0][(size1-window_pt5):(size1+window_pt5)]*v2t1*pzt_coeff, traces1[1][(size1-window_pt5):(size1+window_pt5)], label='intermediate axis')
plt.xlabel('$\mathrm{freq}\,[\mathrm{Hz}]$') 
plt.xlim([(traces2[0][(size2-window_pt5):(size2+window_pt5)]*v2t1*pzt_coeff)[0], (traces2[0][(size2-window_pt5):(size2+window_pt5)]*v2t1*pzt_coeff)[-1]])
plt.ylabel('$\mathrm{P}_\mathrm{REFL}/ \mathrm{P}_\mathrm{max}\,[\mathrm{arb.}]$')
plt.legend()
plt.savefig('../../../../../figs/ALGAAS/split_cav_scan.pdf', dpi=300, format='pdf', bbox_inches='tight')

