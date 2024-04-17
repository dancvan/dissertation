import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib
import pockels_cal as pc

#measpath = '../measurements/swept/algaas/10_18_2021/'

slow_data_path = 'measurements/TF/slow/'
fast_data_path = 'measurements/TF/fast/'

plt.style.available

plt.style.use('ppt2latex')
plt.rcParams["font.family"] = "Times New Roman"


def bode_plt(tf_tuple, save_path, lbl, title, ylbl='dB'):
    ff = tf_tuple[0]
    db = tf_tuple[1]
    deg = tf_tuple[2]
    bode_fig = plt.figure()
    plt.subplot(211)
    if not ylbl=='dB':
        plt.loglog(ff, db, label=lbl)
    else:
        plt.semilogx(ff,db, label = lbl)
    plt.xlim(ff[0], ff[-1])
    plt.ylabel(ylbl)
    plt.legend()
    plt.title(title.replace('_', '\_'))
    plt.subplot(212)
    plt.semilogx(ff,deg, label = lbl)
    plt.xlim(ff[0], ff[-1])
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('phase [deg]')
    plt.savefig(save_path + '/' + title + '.png', dpi=300,bbox_inches='tight')
    plt.close()
    return bode_fig


sdata = pc.tf_import(slow_data_path)
fdata = pc.tf_import(fast_data_path)
f_slow= sdata[0]
f_fast = fdata[0]
db_slow = sdata[1]
db_fast = fdata[1]
deg_slow = sdata[2]
deg_fast = fdata[2]
plt.subplot(211)
plt.semilogx(f_slow,db_slow, label = 'slow\_axis')
plt.semilogx(f_fast,db_fast, label = 'fast\_axis')
plt.xlim(f_slow[0], f_slow[-1])
plt.ylabel('Magnitude [dB]')
plt.legend()
plt.subplot(212)
plt.semilogx(f_slow,deg_slow, label = 'slow\_axis')
plt.semilogx(f_fast,deg_fast, label = 'fast\_axis')
plt.xlim(f_slow[0], f_slow[-1])
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('phase [deg]')
plt.show()

import agilent4395a as ag

f, db, ph = ag.tf_import(measpath, filename='p_pol.TXT')

f2, db2, ph2 = ag.tf_import(measpath, filename='s_p_pol.TXT')

f3, db3, ph3 = ag.tf_import(measpath, filename='s_pol.TXT')

plt.semilogx(f,db)
plt.semilogx(f2,db2)
plt.semilogx(f3,db3)
