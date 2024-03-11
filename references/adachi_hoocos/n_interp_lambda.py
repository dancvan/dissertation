from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""

Interpolating data across 1064 nm (1.166 [eV]) from handbook on optical constants of semiconductors (Adachi 2012)

"""

alas_raw = pd.read_csv('hoocos_alas.csv')                                       # Aluminum Arsenide data
gaas_raw = pd.read_csv('hoocos_gaas.csv')

starteV = .5
endeV = 2.0
deV = 1e-3
eV_rng = np.arange(starteV, endeV, deV)

lambd = 1064e-9

ev = lambda wav : round((4.13566e-15*2.998e8)/wav, 3)

def spline_fit(x, y, arng):
    return splev(arng, splrep(x, y))

#order of the transcribed data by column ('eV' [photon energy], 'eps1' is the real component of the dielectric, 'n' index of refraction, 'R')


## alaas (eV / n plotting raw and spline fit)
alas_fit = spline_fit(alas_raw['eV'], alas_raw['n'], eV_rng)

ind = int((ev(lambd) - starteV) / deV)

print('AlAs refractive index @ ' + str(lambd) + ' is ' + str(alas_fit[ind]))

plt.figure(1)
plt.plot(alas_raw['eV'], alas_raw['n'], 'o', eV_rng, alas_fit)

## gaas (eV / n plotting raw and spline fit)
gaas_fit = spline_fit(gaas_raw['eV'], gaas_raw['n'], eV_rng)


print('GaAs refractive index @ ' + str(lambd) + ' is ' + str(gaas_fit[ind]))

plt.figure(2)
plt.plot(gaas_raw['eV'], gaas_raw['n'], 'o', eV_rng, gaas_fit)

plt.show()

