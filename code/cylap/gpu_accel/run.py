import laplace
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('stylelib/surftex')
from matplotlib import cm
from matplotlib import rcParams
import time 
import torch
from scipy.interpolate import griddata

import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_gpu

## Params import

pdict = laplace.params()

torch.version.cuda

laplace.gpu_check(pdict['gpu_accel'])

roun = lambda val, dec : np.round(int(val)*(10.0**-dec), decimals=dec)

jpu_bool, device = laplace.gpu_check(pdict['gpu_accel'])
if ((pdict['gpu_accel'] and gpu_bool) and pdict['device']!='cpu'):
    print("GPU ready to use with pytorch")
    torch.set_printoptions(precision=8)
    pdict['bitres'] = eval(pdict['bitres'])
    pdict['device'] = device
else:
    print("No GPU available, will resume computaion solely with CPU")


# initialize coordinates
coord_dict = laplace.init_coords(pdict)

# imposing a square simulation space
N = pdict['N'][0]

# coord vecs
rho = np.round(coord_dict['coords']['rho'], decimals=pdict['roi_res_exp'][0])
z = np.round(coord_dict['coords']['z'], decimals=pdict['roi_res_exp'][1])
inv_rho = coord_dict['coords']['invrho']

rho2 = np.round(coord_dict['roi']['rho'], decimals=pdict['roi_res_exp'][0])
z2 = np.round(coord_dict['roi']['z'], decimals=pdict['roi_res_exp'][1])
inv_rho2 = coord_dict['roi']['invrho']

#indices
irho = coord_dict['indices']['rho']
iz = coord_dict['indices']['z']

# Potential map initialization (V) with Dielectric tensor initialization (chi_e)

fV = laplace.anal_sol(pdict)

# intialize potential map, electric susceptibility, and LAMBD operator

V, LAMBD = laplace.init_V(pdict, coord_dict)
chi_e = laplace.init_V(pdict, coord_dict, vec_ind=False)

V_roi = laplace.init_V(pdict, coord_dict, vec_ind=False)
chi_e_roi = laplace.init_V(pdict, coord_dict, vec_ind=False)

chi_e_sub = pdict['optic']['sub_eps']-1
chi_e_coat = pdict['optic']['coat_eps']-1

# Translating (Cauchy) boundary conditions to sim

# Essential boundary condition locations (represented as masks)
fp = 'front_plate'
bp = 'back_plate'

roi_mult_exp = int(abs(np.log10(pdict['roi_mult'])))

fp_thick_bool = np.logical_and(z>=pdict[fp]['zpos'], z<=pdict[fp]['zpos'] + pdict[fp]['thickness'])

bp_thick_bool = np.logical_and(z<=pdict[bp]['zpos'], z>=pdict[bp]['zpos'] - pdict[bp]['thickness'])

# Some lazy functions
maxval = lambda mag : (mag == mag.max()).reshape(pdict['vec_shape'])
minval = lambda mig : (mig == mig.min()).reshape(pdict['vec_shape'])
maxval_min1 = lambda mag, ax : np.roll(mag==mag.max(), -1, axis = ax).reshape(pdict['vec_shape'])
minval_plus1 = lambda mig, ax : np.roll(mig==mig.min(), 1, axis = ax).reshape(pdict['vec_shape'])
plate_geom = lambda _rho : (_rho >= pdict[fp]['hole_diam']/2.0) & (_rho<=pdict[fp]['diam']/2.0)
#foo = np.round(pdict['loc_params']['front of optic']['z'], pdict['res_exp'][1])
foo = np.round((pdict['optic']['thickness']/2.0), pdict['res_exp'][1]+roi_mult_exp)
ctb = np.round((pdict['optic']['diam']/2.0), pdict['res_exp'][0]+roi_mult_exp)
sub = lambda _rho, _z : (((_z>pdict['optic']['z_com']-foo)&(_z<(np.abs(pdict['optic']['z_com'])+foo))) & (_rho <= ctb)).reshape(pdict['vec_shape'])
coat = lambda _rho, _z : ((_z >= foo+pdict['optic']['z_com']) & (_z <= foo + pdict['optic']['z_com'] + pdict['optic']['coat_thickness']) & (_rho <= ctb)).reshape(pdict['vec_shape'])

bc_mask ={'edge' : 
           {'c1_max' : maxval(rho),
            'c1_min' : minval(rho),
            'c2_max' : maxval(z),
            'c2_min' : minval(z)},
          'expo':          # exponential bcs
           {'c1_end' : maxval_min1(rho, 0),
            'c1_0' : minval_plus1(rho, 0),
            'c2_end' : maxval_min1(z, 1),
            'c2_0' : minval_plus1(z, 1)},
          'electrodes':    # capacitor plates
           { fp : (plate_geom(rho) & fp_thick_bool).reshape(pdict['vec_shape']),
             bp : (plate_geom(rho) & bp_thick_bool).reshape(pdict['vec_shape'])},
          'optic':        # substrate
           {'sub' : sub(rho, z),
            'coat1' : ((z==np.round(pdict['loc_params']['front of optic']['z'],pdict['res_exp'][1])) & (rho <= np.round((pdict['optic']['diam']/2), pdict['res_exp'][0]))).reshape(pdict['vec_shape']),
            'coat2' : ((z==np.round((pdict['loc_params']['front of optic']['z']-pdict['res'][1]),pdict['res_exp'][1])) & (rho <= (np.round(pdict['optic']['diam']/2, pdict['res_exp'][0])))).reshape(pdict['vec_shape'])}
          }
roi_mask ={'edge':
           {'c1_max' : maxval(rho2),
            'c1_min' : minval(rho2),
            'c2_max' : maxval(z2),
            'c2_min' : minval(z2)},
           'electrodes':
           { fp : (plate_geom(rho2) & (z2 == pdict[fp]['zpos'])).reshape(pdict['vec_shape']),
             bp :(plate_geom(rho2) & (z2 == pdict[bp]['zpos'])).reshape(pdict['vec_shape'])},
           'optic' : 
           {'sub' : sub(rho2, z2),
            'coat' : coat(rho2, z2)
           }
          }  


## Initialize BCs

#### Set susceptibility
chi_e[bc_mask['optic']['sub']] = chi_e_sub
#chi_e[bc_mask['optic']['coat1']] = chi_e_coat
#chi_e[bc_mask['optic']['coat2']] = chi_e_coat


chi_e_roi[roi_mask['optic']['sub']] = chi_e_sub
chi_e_roi[roi_mask['optic']['coat']] = chi_e_coat

## Electro-static conditions

### Electrode plates
V[bc_mask['electrodes'][fp]] = pdict[fp]['voltage']
V[bc_mask['electrodes'][bp]] = pdict[bp]['voltage']


def apply_exp(V, pdict, mask, V_char=0, R_char=1.0):
    V_exp = lambda V_0, R_0, V, R : V_0 + np.exp(-R/R_0)*(V-V_0)
    ### Boundary values
    #rho=rho_max    
    V[mask['edge']['c1_max']] = V_exp(V_char, R_char, V[mask['expo']['c1_end']], pdict['res'][0])
    #z=z_min
    V[mask['edge']['c2_min']] = V_exp(V_char, R_char, V[mask['expo']['c2_0']], pdict['res'][1])
    #z=z_max
    V[mask['edge']['c2_max']] = V_exp(V_char, R_char, V[mask['expo']['c2_end']], pdict['res'][1])
    
    

def apply_dirichlet(V, pdict, mask):
    ### Electrode plates
    V[mask['electrodes']['front_plate']] = pdict['front_plate']['voltage']
    V[mask['electrodes']['back_plate']] = pdict['back_plate']['voltage']
    
apply_exp(V, pdict, bc_mask, V_char=0, R_char=1.0)
apply_dirichlet(V, pdict, bc_mask)

#### Edge for faster convergance
V[bc_mask['edge']['c2_max']] = pdict[fp]['voltage']
V[bc_mask['edge']['c2_min']] = pdict[bp]['voltage']

if pdict['gpu_accel']:
    V[bc_mask['edge']['c1_max']] = torch.from_numpy(np.interp(np.arange(0,pdict['N'][0]), np.array([0,pdict['N'][0]-1]), np.array([pdict[bp]['voltage'], pdict[fp]['voltage']]))).type(torch.float32)
else:
    V[bc_mask['edge']['c1_max']] = np.interp(np.arange(0,pdict['N'][0]), np.array([0,pdict['N'][0]-1]), np.array([pdict[bp]['voltage'], pdict[fp]['voltage']]))


# Plotting persistent BCs
figstr = int(1e-2/pdict['grid_res'])
fig = plt.figure(figsize = (22,11))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(rho, z, V.reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
#ax.plot_surface(rho, z, bc_mask['expo']['c2_end'].reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
#ax.plot_surface(rho, z, bc_mask['expo']['c2_0'].reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
#ax.plot_surface(rho, z, bc_mask['expo']['c1_end'].reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
ax.set_xlabel('r [m]')
ax.set_ylabel('z [m]')
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(rho, z, chi_e.reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
ax2.set_xlabel('r [m]')
ax2.set_ylabel('z [m]')
#ax3 = fig.add_subplot(2,2,3, projection='3d') 
#ax3.plot_surface(rho, z, chi_e.reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
#ax3.set_xlabel('r [m]')
#ax3.set_ylabel('z [m]')
#ax4 = fig.add_subplot(2,2,4, projection='3d') 
#ax4.plot_surface(rho, z, V.reshape(N,N),rstride=figstr,cstride=figstr,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
#ax4.set_xlabel('r [m]')
#ax4.set_ylabel('z [m]')
plt.tight_layout()

#### Build operators
lap = laplace.build_lap(pdict, LAMBD, irho)
grad = laplace.build_grad(pdict, LAMBD)
disp = laplace.build_disp(pdict, LAMBD)
LAP = laplace.build_LAP(pdict, coord_dict['coords'], lap, grad, disp, chi_e)
if pdict['gpu_accel'] and (pdict['device']=='cuda') and pdict['torch']:
    LAP = LAP.to(device=pdict['device'])
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

torch.cuda.empty_cache()

t = time.time()
#### run sim

if pdict['gpu_accel'] and pdict['torch']!=True:
    iter_const = cp.array((pdict['res'][0]**2)*pdict['iter_step'])
    V_gpu = cp.array(V)
    for itr in range(0, pdict['iters']):
        V_gpu = V_gpu + LAP.dot(V_gpu*iter_const)
        ## Re-applying BCs
        apply_exp(V_gpu, pdict, bc_mask)
        apply_dirichlet(V_gpu, pdict, bc_mask)
        torch.cuda.empty_cache()
elif (pdict['gpu_accel'] and (pdict['device']=='cuda') and pdict['torch']) or pdict['device'] == 'mps':
    V = V.to(device=pdict['device'])
    iter_const = 9*torch.tensor((10**(np.log10(pdict['res'][0])*2 + np.log10(pdict['iter_step']))), dtype=torch.float64, device='cuda')
    for itr in range(0, pdict['iters']):
        V = V + torch.sparse.m(LAP,V).mul(iter_const)
        apply_exp(V, pdict, bc_mask)
        apply_dirichlet(V, pdict, bc_mask)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
else: 
    iter_const = pdict['res'][0]**2*pdict['iter_step']
    for itr in range(0, pdict['iters']):
        V = V + LAP*V*iter_const
        apply_exp(V, pdict, bc_mask)
        apply_dirichlet(V, pdict, bc_mask)
        
elapsed = time.time() - t
print(elapsed)

V_gpu.data

V_gpu

fig = plt.figure(figsize = (18.5,21))
ax = plt.axes(projection='3d') 
surf = ax.plot_surface(rho, z, V.reshape(N,N),rstride=1,cstride=1,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
fig.tight_layout()
ax.view_init(20,210)
ax.set_xlabel('r [m]')
ax.set_ylabel('z [m]')
ax.set_zlabel('[V]')
fig.colorbar(surf, shrink=0.4, aspect=20, pad=-0.025)
axes_width = fig.get_size_inches()[1]*(fig.subplotpars.right-fig.subplotpars.left)
right =1.095
left =-.15
fig.subplots_adjust(left=left,right=right) 
fig.set_size_inches((fig.get_size_inches()[0],axes_width/(right-left)))
ax.tick_params(axis='both', pad=15)
axes_height = fig.get_size_inches()[1]*(fig.subplotpars.top-fig.subplotpars.bottom)
top = 1.15
bottom=-.09
fig.subplots_adjust(top=top,bottom=bottom)             
fig.set_size_inches((fig.get_size_inches()[0],axes_height/(top-bottom)))
#fig.savefig(fig_exp_dir + 'assembly1_sim.pdf', dpi=300, format='pdf')
