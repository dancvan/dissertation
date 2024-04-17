import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib
import pockels_cal as pc

#measpath = '../measurements/swept/algaas/10_18_2021/'
expdir = '../../../figs/ALGAAS/'

slow_data_path = 'slow/'
fast_data_path = 'fast/'

plt.style.available

plt.style.use('ppt2latex')
plt.rcParams["font.family"] = "Times New Roman"

sdata = pc.tf_import(slow_data_path)
fdata = pc.tf_import(fast_data_path)
f_slow= sdata[0]
f_fast = fdata[0]
db_slow = sdata[1]
db_fast = fdata[1]
deg_slow = sdata[2]
deg_fast = fdata[2]
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.semilogx(f_slow,db_slow, label = 'slow\_axis')
ax1.semilogx(f_fast,db_fast, label = 'fast\_axis')
ax1.set_ylabel('Magnitude [dB]')
ax1.legend()
ax2.semilogx(f_slow,deg_slow, label = 'slow\_axis')
ax2.semilogx(f_fast,deg_fast, label = 'fast\_axis')
ax2.set_xlim(f_slow[0], f_slow[-1])
ax2.legend()
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [deg]')
#plt.show()
fig.savefig(expdir + 'rawtf_fast_slow.pdf', dpi=300, bbox_inches='tight')

import agilent4395a as ag

f, db, ph = ag.tf_import(measpath, filename='p_pol.TXT')

f2, db2, ph2 = ag.tf_import(measpath, filename='s_p_pol.TXT')

f3, db3, ph3 = ag.tf_import(measpath, filename='s_pol.TXT')

plt.semilogx(f,db)
plt.semilogx(f2,db2)
plt.semilogx(f3,db3)
