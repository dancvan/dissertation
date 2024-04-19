import numpy as np
import matplotlib.pyplot as plt

## import both tfs
#SCRN0530.TXT
#SCRN0531.TXT
#SCRN0534.TXT
#SCRN0535.TXT

magmeas1 = np.loadtxt('SCRN0530.TXT').transpose()
phmeas1 =  np.loadtxt('SCRN0531.TXT').transpose()
magmeas2 = np.loadtxt('SCRN0534.TXT').transpose()
phmeas2 =  np.loadtxt('SCRN0535.TXT').transpose()

plt.style.use('ppt2latex')
plt.rcParams["font.family"] = "Times New Roman"

def bode_plt(tf_tuple, save_path, lbl, ylbl='dB'):
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
    plt.subplot(212)
    plt.semilogx(ff,deg, label = lbl)
    plt.xlim(ff[0], ff[-1])
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('phase [deg]')
    plt.savefig(save_path + '/' + lbl + '.pdf', dpi=300,bbox_inches='tight')
    plt.close()
    return bode_fig

## import both tfs
#trek_mag.TXT
#trek_phase.TXT
#trekmag_10to100.TXT
#trekphase_10to100.TXT

meastr1 = [magmeas1[0], magmeas1[1], phmeas1[1]]
meastr2 = [magmeas2[0], magmeas2[1], phmeas2[1]]

## put both on single bode plt

bode_plt(meastr1, save_path='.', lbl='tf1')
bode_plt(meastr2, save_path='.', lbl='tf2')

