import numpy as np

# some constants:
cee = np.float64(299792458)
h_bar = (6.626e-34)/(2*np.pi)


# Michelson frequency response
def mich_freq_resp(freq, Length, phi_0, P_in):
    """ 
    freq : standard frequency [Hz]
    Length : Michelson ifo arm length [m]
    phi_0 : microscopic differential arm tuning [rad]
    P_in : input power [W]
    """
    return (P_in*cee*np.sin(phi_0))*Length*np.exp((-1j*Length*2.0*np.pi*freq)/cee)*np.sin((Length*2.0*np.pi*freq)/cee)/(Length*2.0*np.pi*freq)

# Shot noise
def N_shot(OMEG, Length, phi_0, P_in):
    """
    Interferometer shot noise calculator
    Inputs:
    OMEG: OPTICAL angular frequency [rad Hz]
    Length : ifo arm length [m]
    phi_0 : microscopic differential arm tuning [rad]
    P_in : Input power [W]
    """
    return 2*np.sqrt(2*h_bar*OMEG*(2 - 2*np.cos(phi_0)))