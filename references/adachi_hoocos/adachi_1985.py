import numpy as np
import matplotlib.pyplot as plt
"""
Computing the real part of the dielectric constant for varying aluminum composition (0 <= x <= 1)

"""


lambd = 1064e-9                                                                                                         #Chosen wavelength (to be converted to photon energy)
x = 0.92                                                                                                                #Al_x Ga_(1-x) As

h_bar = (4.136e-15)/(2*np.pi)                                                                                           #[eV*m]
cee = 2.998e8                                                                                                           #[m/s]

omega = (2.0*np.pi*cee)/lambd                                                                                           #natural frequency

E0 = lambda x_Al : 1.724 + 1.247*x_Al + 1.147*(x_Al-.45)**2                                                             #[eV]
del0 = .3                                                                                                               #[eV]

chi_ = lambda E : (h_bar*omega) / E                                                                                     #dimensionless

f_diel_fit = lambda chi : (chi**-2)*(2 - (1.0+chi)**.5 - (1.0-chi)**.5)                                                 #############
                                                                                                                        #############
A0 = lambda x_Al : 6.3 + 19.0*x_Al                                                                                      #                                                                               
                                                                                                                        #
B0 = lambda x_Al : 9.4 - 10.2*x_Al                                                                                      # Adachi 1985
                                                                                                                        #
#chi_E0 = chi_(E0)                                                                                                      #
#chi_E0_del0 = chi_(E0 + del0)                                                                                          #   
                                                                                                                        #############
eps1 = A0(x) * (f_diel_fit(chi_(E0(x))) + .5*((E0(x) /(E0(x) + del0))**(3/2)) * f_diel_fit(chi_(E0(x) + del0))) + B0(x) #############

print('The refractive index estimate estimate for Al_' + str(x) + 'Ga_' + str(round(1.0-x, 2)) + 'As ' + 'is ' + str((eps1)**.5))
