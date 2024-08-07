# Interferometer configurations## MICH

```python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0,'../')
plt_style_dir = '../../stash/'
fig_exp_dir = '../../../figs/'
from ifo_configs import N_shot
from ifo_configs import mich_freq_resp as MICH
from ifo_configs import bode_amp, bode_ph
%matplotlib inline
if os.path.isdir(plt_style_dir) == True:
    plt.style.use(plt_style_dir + 'ppt2latexsubfig.mplstyle')
plt.rcParams["font.family"] = "Times New Roman"
```


```python
# Some parameters
cee = np.float64(299792458)
h_bar = (6.626e-34)/(2*np.pi)
OMEG = np.float64(2*np.pi*cee/(1064.0*1e-9))
L = np.float64(4000.0)
nu = np.arange(1, 1000000, 1)
PHI_0 = np.pi/2 #[rad]
P_IN = 125 #[W]
```
### Derivation

For the simple Michelson we know that a change in arm length correlates to light at the AS port
We also know that a differential arm length corresponds to a difference in phase of the light that impinges upon the BS
For a gravitational wave we can quantify the phase difference in this following way: 

$$
\phi_A - \phi_B = \int_{t-2L/c}^{t} \Omega \bigg[1 + \frac{1}{2}h(t)\bigg]dt - \int_{t-2L/c}^{t} \Omega \bigg[1 - \frac{1}{2}h(t)\bigg]dt \label{eq1}\tag{1}
$$
The phase difference can then be quantified by:
$$
\phi_A - \phi_B = \int_{t-2L/c}^{t} \Omega h(t)dt \label{eq2}\tag{2}
$$
where 
$$ 
h(t) = h_0 e^{i \omega t} \label{eq3}\tag{3}
$$

*$\Omega$* is the **optical angular frequency**

After evaluating this integral we get: 
$$
\Delta \phi=\phi_A - \phi_B = \frac{2 L \Omega}{c}e^{-i L \omega / c} \frac{\mathrm{sin}(L \omega /c)}{L \omega /c} \cdot h_0 e^{i \omega t}
$$

Where the first term in the phase difference carries all the time independent frequency information. This is what we are calculating below. 

For the sake of being explicit, we are going to plot: 
$$
\Delta \phi (\omega) = h_0\frac{2 L \Omega}{c}e^{-i L \omega / c} \frac{\mathrm{sin}(L \omega /c)}{L \omega /c}
$$This accounts for the differential phase as a function of gravitational wave frequency, though we have not established the amount of optical gain the Michelson offers. This can be understood through a first order taylor approximation about a selected Michelson offset angle $\phi_0$:

$$P(\omega, \phi_0) =  \frac{P_\mathrm{in}}{4} [r_x^2 + r_y^2 -  2r_x r_y\mathrm{cos}(\phi_0 + \Delta \phi (\omega)] $$

$$P(\omega, \phi_0) \approx  \frac{P_\mathrm{in}}{4} \Big[ r_x^2 + r_y^2 -  2r_x r_y \big(\mathrm{cos}(\phi_0) - \Delta \phi(\omega) \cdot \mathrm{sin}(\phi_0) \big) \Big] =  \frac{P_\mathrm{in}}{2} \Big[1 - \big(\mathrm{cos}(\phi_0) - \Delta \phi(\omega) \cdot \mathrm{sin}(\phi_0) \big) \Big]$$

Where we define a response gain function $H_\mathrm{MICH}$:

$$\mathrm{H}_\mathrm{MICH}(\omega, \phi_0) =   \frac{P_\mathrm{in}}{2} \cdot \Delta \phi(\omega) \cdot \mathrm{sin}(\phi_0)$$

```python
H = MICH(nu, L, PHI_0, P_IN, OMEG)
```


```python
fig, ax1 = plt.subplots()
ax1.set_xlabel('frequency [Hz]')
ax1.set_ylabel('H$_{\mathdefault{MICH}}$ [$\mathdefault{W/m}$]',color='C0')
#ax1.plot(w/(FSR), F_w_cc_modsq*100)
ax1.loglog(bode_amp(H),linewidth=7.5, color='C0')
#plt.ylim([10e-6, 10e0])
ax2 = ax1.twinx()
#ax2.plot(w/(FSR), (180/np.pi)*np.arctan(F_w_cc.imag/F_w_cc.real), '--')
ax2.semilogx(nu,(180/np.pi)*np.arctan(np.imag(H)/np.real(H)), '--', linewidth=7.5,color='C1')
#plt.xlabel('frequency [FSR]')
plt.xlim([1,1e5])
plt.ylabel('phase [deg]',color='C1')
fig.savefig(fig_exp_dir + 'INTRO/mich_fr.pdf', dpi=300, bbox_inches='tight')
```


    
![png](mich_fr_raw_files/mich_fr_raw_7_0.png)
    

Though with the provided frequency depdenence and optical gain, we still need to understand a starting noise floor spectra and compare to our anticipated
* The noise floor is established## Interferometer noise and sensitivity### Shot noise
* A fundamental limit imposed by the statistical nature of photon counting
* The photon counting follows Poisson statistics
    * Photon counting variance (variance is equal to the mean)
$$ < (n-\bar{n})^2 >  = \frac{P \Delta t}{ \hbar \Omega} $$
    * Power variance:
$$ < (P - \bar{P})^2 >  = \hbar \Omega  \bar{P} \Delta t $$
    * PSD of the measured power between two uncoorelated moments in time:
$$ S_\mathrm{P} (\omega) = \lim_{T \to \infty} \frac{2}{T} \Big< \big| \int_{-T}^{T} (P(t) - \bar{P}) e^{-i\omega t} dt \big|^2 \Big> $$
$$ =  \lim_{T \to \infty} \frac{2}{T} \int_{-T}^{T} \hbar \Omega \bar{P} dt  $$
$$ = 2 \hbar \Omega \bar{P} $$
    * Where the ASD is:
$$ [S_P (\omega)]^{1/2} = [2 \hbar \Omega \bar{P}]^{1/2}$$The signal to noise is established by dividing the frequency dependent optical gain times the gravitational wave ASD $\big( [\mathrm{S}_{\mathrm{h}}(\Omega)]^{1/2} \big)$ by the noise ASD:

$$\mathrm{SNR} = \mathrm{G_{opt}(\omega)} [\mathrm{S}_{\mathrm{h}}(\omega)]^{1/2} / S_\mathrm{N}(\omega) = \mathrm{H}_\mathrm{MICH} / [S_P]^{1/2} = \bigg( \frac{\Delta \phi(\omega)}{h_0} \frac{P_\mathrm{in}}{2}\mathrm{sin}(\phi_0) \bigg) \bigg/ [2 \hbar \Omega \bar{P}]^{1/2}$$

This is to say that for the stated gravitational wave ASD, and for an SNR of 1, we establish the following threshold for detector:

$$\big[ \mathrm{S}_{\mathrm{h}}(\omega) \big]^{1/2} \; \{\mathrm{SNR}\geq1\} \geq \frac{ [S_\mathrm{N}(\omega)]^{1/2}}{\mathrm{H}_\mathrm{MICH}(\omega)}$$

Where 

$$\frac{ [S_\mathrm{N}(\omega)]^{1/2}}{\mathrm{H}_\mathrm{MICH}(\omega)} = \frac{[2 \hbar \omega \bar{P}]^{1/2}}{ \Delta \phi(\omega) [P_\mathrm{in} / 2]  \mathrm{sin}(\phi_0)} = \bigg( \frac{\hbar \Omega }{\omega P_\mathrm{in}} \bigg)^{1/2} \frac{[r_x^2 + r_y^2 -  2r_x r_y\mathrm{cos}(\phi_0)]^{1/2}}{\mathrm{sin}(L \omega / c)} e^{iL \omega / c}$$

```python
S_h = N_shot(OMEG, P_IN) 
print(S_h)
```

    6.831801787605637e-09



```python
#ax1.plot(w/(FSR), F_w_cc_modsq*100)
plt.loglog(nu, S_h/bode_amp(H), linewidth=7.5, color='C0')
plt.ylim([1e-21, .5e-14])
plt.xlabel('frequency [Hz]')
plt.ylabel('$\mathdefault{S}_\mathdefault{h} \;  \mathdefault{[ 1 / \sqrt{\mathdefault{Hz}}]} $')
#ax2_ = ax1_.twinx()
#ax2.plot(w/(FSR), (180/np.pi)*np.arctan(F_w_cc.imag/F_w_cc.real), '--')
#ax2_.semilogx(nu,(180/np.pi)*np.arctan(np.imag(S_h)/np.real(S_h)), '--', linewidth=7.5,color='C1')
#plt.xlabel('frequency [FSR]')
plt.xlim([1,1e5])
plt.grid(visible=True)
#plt.subplots_adjust(hspace = 1)
#plt.ylabel('phase [deg]',color='C1')
#plt.tight_layout(rect=[0,0,1,1])
#plt.title('')
#plt.subplots_adjust(bottom=.1, top=.85) #, right=.8, left=.1)
plt.savefig(fig_exp_dir + 'INTRO/mich_sensi.pdf', dpi=300, bbox_inches='tight')
```


    
![png](mich_fr_raw_files/mich_fr_raw_13_0.png)
    

