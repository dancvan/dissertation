import numpy as np

# Bode tools 
def bode_amp(H):
    """
    Returns amplitude information on transfer function (H)
    """
    return np.sqrt(np.real(H)**2 + np.imag(H)**2)

def bode_ph(H):
    """
    Returns phase information on transfer function (H)
    """
    return (180/np.pi)*np.arctan(np.imag(H)/np.real(H))

# some constants:
cee = np.float64(299792458) ## speed of light [m/s]
h_bar = (6.626e-34)/(2*np.pi) ## planck's constant


# IFO params
def finesse(r_i, r_e):
    """
    r_i : ITM reflectivity coefficient
    r_e : ETM reflectivity coefficient
    """
    return np.pi*np.sqrt(r_i*r_e)/(1-(r_i*r_e))


# Michelson frequency response
def mich_freq_resp(freq, Length, phi_0, P_in, OMEGA):
    """ 
    MICHELSON FREQEUNCY RESPONSE CALCULATOR
    freq : standard (gravitational wave) frequency [Hz]
    Length : Michelson ifo arm length [m]
    phi_0 : static differential arm length tuning phase [rad]
    P_in : input power [W] 
    """
    return (P_in*OMEGA*np.sin(phi_0))*Length*np.exp((-1j*Length*2.0*np.pi*freq)/cee)*np.sin((Length*2.0*np.pi*freq)/cee)/(Length*2.0*np.pi*freq)

def fpmi_freq_resp(freq, r_1, t_1, r_2, L, phi_0, P_in, OMEGA, low_pass=False):
    """
    FABRY PEROT MICHELSON FREQUENCY RESPONSE CALCULATOR
    freq : standard (gravitational wave) frequency [Hz]
    r_1, t_1, r_2: Assuming arm symmetry where the ITM has r_1, t_1 coefficients and the ETM has a r_2 reflectivity coefficient. Also assumes no loss. [arb]
    OMEGA: OPTICAL angular frequency [rad Hz]
    Length: Michelson ifo arm length [m]
    phi_0 : static differential arm length tuning phase [rad]
    """
    if low_pass:
        f_pole = 1/(((4*np.pi*L)*np.sqrt(r_1*r_2))/(cee*(1-r_1*r_2)))
        fpmi_resp = 1/(1 + 1j*(freq/f_pole))
    else:
        fpmi_resp = ((t_1**2 * r_2)/((t_1**2 + r_1**2)*r_2 - r_1))*(mich_freq_resp(freq, L, phi_0, P_in, OMEGA)/(1-r_1*r_2*np.exp(-1j*L*4.0*np.pi*freq/cee)))
    return fpmi_resp

def PRG(L_rt, Finn, r_PRM, max=0):
    """
    POWER RECYCLING GAIN (@ optimal reflectivity)
    * Assuming a FPMI with symmetric arms *
    L_rt : Round trip loss
    Finn : Cavity finesse
    """
    if max == 1:
        G_PR = np.pi/(2*Finn*L_rt*(1-((Finn*L_rt)/(2*np.pi))))
    else:
        G_PR = (1-r_PRM**2)/(1-r_PRM*(1-(Finn/np.pi)*L_rt))**2


    return G_PR

def drfpmi_freq_resp(freq, G_PRC_opt, r_1, t_1, r_2, r_SRM, t_SRM, phi_SRC, L, phi_0, P_in, OMEGA):
    """
    DUAL RECYCLED FABRY PEROT MICHELSON FREQUENCY RESPONSE CALCULATOR

    %%%$$\mathrm{H}_\mathrm{DRFPMI} = \mathrm{G}_\mathrm{PR} \mathrm{P}_\mathrm{in} L \Omega \bigg[ \frac{ t_1^2 r_2}{(t_1^2 + r_1^2)r_2 - r_1} \frac{t_\mathrm{SRM} t_1 e^{-i\phi_\mathrm{SRC}}}{1-r_1 r_\mathrm{SRM} e^{-i2\phi_\mathrm{SRC}}} \frac{e^{-i 2 \pi L f / c} \mathrm{sin}( 2 \pi f / c)}{ 2 \pi L f } \frac{\mathrm{sin}(\phi_0)}{1- [(r_1 - r_\mathrm{SRM} e^{-i2\phi_\mathrm{SRC}})/(1-r_1 r_\mathrm{SRM}e^{-i2\phi_\mathrm{SRC}})] r_2 e^{-i 4 \pi L f / c}} \bigg]$$%%%

    freq: standard (gravitational wave) frequency [Hz]
    G_PRC_opt: maximum power recycling gain (optimal) [arb]
    r_1: ITM reflection coefficient [arb]
    t_1: ITM transmission coefficient [arb]
    r_2: ETM reflection coefficient [arb]
    r_SRM: Signal recycling mirror reflection coefficient [arb]
    t_SRM: Signal recycling mirror transmission coefficient [arb]
    L: Length of the Fabry-Perot arms [m]
    OMEGA: OPTICAL angular frequency [rad Hz]
    """
    r_SRC = (r_1 - r_SRM*np.exp(1j*2*phi_SRC))/(1 - r_1*r_SRM*np.exp(1j*2*phi_SRC))
    t_SRC = t_1*t_SRM*np.exp(1j*phi_SRC)/(1 - r_1*r_SRM*np.exp(1j*2*phi_SRC))

    return ((t_1**2 * r_2)/((t_1**2 + r_1**2)*r_2 - r_1))*G_PRC_opt*t_SRC*(P_in*L*OMEGA*np.exp((-1j*L*2.0*np.pi*freq)/cee)*np.sin((L*2.0*np.pi*freq)/cee)/(L*2.0*np.pi*freq))/(1-r_SRC*r_2*np.exp(-1j*L*4.0*np.pi*freq/cee))


# Shot noise
def N_shot(OMEGA, P_in):
    """
    Interferometer shot noise calculator
    OMEG: OPTICAL angular frequency [rad Hz]
    Length : ifo arm length [m]
    phi_0 : static differential arm length tuning phase [rad]
    P_in : Input power [W]
    """
    return np.sqrt(2*h_bar*OMEGA*P_in)