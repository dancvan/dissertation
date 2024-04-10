import set_params
import laplace
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('stylelib/surftex')
fig_exp_dir = "../../../../dissertation/figs/ALGAAS/"
from matplotlib import cm
from matplotlib import rcParams
import time

###

pdict = set_params.pdict
pdict

###

pdict['res_exp']

###

# initialize coordinates
coord_dict = laplace.init_coords(pdict) 

# Imposing a square simulation space
N = pdict['N'][0]

# coord vecs
rho = coord_dict['coords']['rho']
z = coord_dict['coords']['z']
inv_rho = coord_dict['coords']['invrho']

#indices 
irho = coord_dict['indices']['rho']
iz = coord_dict['indices']['z']

###

fV = laplace.anal_sol(pdict)
fV['V_anal'](coord_dict['indices']['z']*pdict['res'][1])

###

plt.plot(coord_dict['indices']['z']*pdict['res'][1],fV['V_anal'](coord_dict['indices']['z']*pdict['res'][1]))

### 

coord_dict['indices']['z']*pdict['res'][1]

###

# intialize potential map, electric susceptibility, and LAMBD operator
V = laplace.init_V(N)
chi_e = laplace.init_V(N)
chi_e_sub = pdict['optic']['sub_eps']-1
chi_e_coat = pdict['optic']['coat_eps']-1
LAMBD = laplace.build_lambd(irho, iz, N)

###

# Translating (Dirichlet) boundary conditions to sim 

# Initial value 

## (For faster convergence) setting edge values

### Edge locations
r_max = (rho == max(rho))
r_min = (rho == min(rho))
z_max = (z == max(z))
z_min = (z == min(z))

# Boundary values 

## Plate potentials
fp = 'front_plate'
bp = 'back_plate'

fp_thick_bool = np.logical_and(z>=pdict[fp]['zpos'], z<=pdict[fp]['zpos'] + pdict[fp]['thickness'])

bp_thick_bool = np.logical_and(z<=pdict[bp]['zpos'], z>=pdict[bp]['zpos'] - pdict[bp]['thickness'])

loc_fp = np.logical_and(np.logical_and(rho>=pdict[fp]['hole_diam']/2,rho<=pdict[fp]['diam']/2),fp_thick_bool)
loc_bp = np.logical_and(np.logical_and(rho>=pdict[bp]['hole_diam']/2,rho<=pdict[bp]['diam']/2),bp_thick_bool)
#bc_fp = laplace.BC_dict([[pdict[fp]['hole_diam']/2, pdict[fp]['diam']/2], pdict[fp]['zpos']],pdict[fp]['voltage'],fp, LAMBD)
#bc_bp = laplace.BC_dict([[pdict[bp]['hole_diam']/2, pdict[bp]['diam']/2], pdict[bp]['zpos']],pdict[fp]['voltage'],bp, LAMBD)

# Exponential boundary conditions
exp_rend  = rho==np.round(max(rho)-pdict['res'][0], pdict['res_exp'][0])

exp_z0 = z==np.round(min(z)+pdict['res'][1], pdict['res_exp'][1])

exp_zend = z==np.round(max(z)-pdict['res'][1], pdict['res_exp'][1])

# Setting sample dielectric
loc_sub = np.logical_and(np.abs(z - pdict['optic']['z_com']) < np.round((pdict['optic']['thickness']/2),pdict['res_exp'][1]), (rho<np.round((pdict['optic']['diam']/2),pdict['res_exp'][0])))
loc_coat1 = np.logical_and((z==np.round(pdict['loc_params']['front of optic']['z'],pdict['res_exp'][1])), (rho < np.round((pdict['optic']['diam']/2), pdict['res_exp'][0])))
loc_coat2 = np.logical_and((z==np.round((pdict['loc_params']['front of optic']['z']-pdict['res'][1]),pdict['res_exp'][1])), (rho < (np.round(pdict['optic']['diam']/2, pdict['res_exp'][0]))))

###

## Initialize BCs

#### Set susceptibility
chi_e[loc_sub] = chi_e_sub
chi_e[loc_coat1] = chi_e_coat
chi_e[loc_coat2] = chi_e_coat

## Electro-static conditions

### Electrode plates
V[loc_fp] = pdict[fp]['voltage']
V[loc_bp] = pdict[bp]['voltage']

#### Boundary values

#### Edge for faster convergance
V[z_min] = pdict['back_plate']['voltage']
V[z_max] = pdict['front_plate']['voltage']
V[r_max] = np.interp(np.arange(0,pdict['N'][0]), np.array([0,pdict['N'][0]-1]), np.array([pdict['back_plate']['voltage'], pdict['front_plate']['voltage']]))

### Exponential (Dirichlet) boundary conditions
V_exp = lambda V_0, R_0, V , R : V_0 + np.exp(-R/R_0)*(V - V_0) 
V_char = 0
R_char = 1.0

#rho=rho_max
V[r_max] = V_exp(V_char, R_char, V[exp_rend], pdict['res'][0])
#z=z_min
V[z_min] = V_exp(V_char, R_char, V[exp_z0], pdict['res'][1])
#z=z_max
V[z_max] = V_exp(V_char, R_char, V[exp_zend], pdict['res'][1])

###

fig = plt.figure(figsize = (22,11))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(rho.reshape(N,N), z.reshape(N,N), V.reshape(N,N),rstride=1,cstride=1,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)

ax.set_xlabel('r [m]')
ax.set_ylabel('z [m]')
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(rho.reshape(N,N), z.reshape(N,N), chi_e.reshape(N,N),rstride=1,cstride=1,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
ax2.set_xlabel('r [m]')
ax2.set_ylabel('z [m]')

plt.tight_layout()


#### Build operators
lap = laplace.build_lap(pdict, LAMBD, irho)
grad = laplace.build_grad(pdict, LAMBD)
disp = laplace.build_disp(pdict, LAMBD)
LAP = laplace.build_LAP(pdict, coord_dict, lap, grad, disp, chi_e)

LAP

###

t = time.time() 
#### run sim
for itr in range(0, pdict['iters']):
    
    V = V + (pdict['res'][0]*pdict['res'][0]*pdict['iter_step']*LAP.dot(V))
    
    ## Re-applying exponential condition
    #rho=rho_max
    V[r_max] = V_exp(V_char, R_char, V[exp_rend], pdict['res'][0])
    #z=z_min
    V[z_min] = V_exp(V_char, R_char, V[exp_z0], pdict['res'][1])
    #z=z_max
    V[z_max] = V_exp(V_char, R_char, V[exp_zend], pdict['res'][1])
    
    ### Re-apply Electro-static condition
    V[loc_fp] = pdict[fp]['voltage']
    V[loc_bp] = pdict[bp]['voltage']
elapsed = time.time() - t
print(elapsed)

### 

#z_halfway , V_opt_half = laplace.grabxsect(pdict['loc_params']['halfway out on optic'], coord_dict, V)
z_halfway , V_opt_half = laplace.grabxsect(pdict['loc_params']['center of optic'], coord_dict,pdict['res_exp'], V)
#figpt5 = 
plt.plot(coord_dict['indices']['z']*1e-4, fV['V_anal'](coord_dict['indices']['z']*1e-4), z_halfway, V_opt_half)
plt.plot(z_halfway, V_opt_half)

###

fig1 = plt.figure(figsize = (18.5,21))
ax = plt.axes(projection='3d') 
surf = ax.plot_surface(rho.reshape(N,N), z.reshape(N,N),V.reshape(N,N),rstride=1,cstride=1,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
fig1.tight_layout()
ax.view_init(20,210)
ax.set_xlabel('r [m]')
ax.set_ylabel('z [m]')
ax.set_zlabel('[V]')
fig1.colorbar(surf, shrink=0.4, aspect=20, pad=-0.025)
axes_width = fig1.get_size_inches()[1]*(fig1.subplotpars.right-fig1.subplotpars.left)
right =1.095
left =-.15
fig1.subplots_adjust(left=left,right=right) 
fig1.set_size_inches((fig1.get_size_inches()[0],axes_width/(right-left)))
ax.tick_params(axis='both', pad=15)
axes_height = fig1.get_size_inches()[1]*(fig1.subplotpars.top-fig1.subplotpars.bottom)
top = 1.15
bottom=-.09
fig1.subplots_adjust(top=top,bottom=bottom)             
fig1.set_size_inches((fig1.get_size_inches()[0],axes_height/(top-bottom)))
#ax.set_box_aspect(None, zoom=0.91)

###

# Generating Fields
Erho = grad[0]*V
Ez = grad[3]*V

z_rho0 , E_rho0 = laplace.grabxsect(pdict['loc_params']['center of optic'], coord_dict, pdict['res_exp'], Ez)

z_halfway , E_halfway = laplace.grabxsect(pdict['loc_params']['halfway out on optic'], coord_dict, pdict['res_exp'], Ez)

rho_front , E_front = laplace.grabxsect(pdict['loc_params']['front of optic'], coord_dict, pdict['res_exp'], Ez)

z_halfway

#fig1.pt5 = 
plt.plot(z_halfway, E_halfway, z_rho0, E_rho0) 

E_halfway.min()

fig2 = plt.figure(figsize = (18.5,21))
ax2 = plt.axes(projection='3d') 
surf2 = ax2.plot_surface(rho.reshape(N,N), z.reshape(N,N),Ez.reshape(N,N),rstride=1,cstride=1,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
fig2.tight_layout()
ax2.view_init(20,210)
ax2.set_xlabel('r [m]')
ax2.set_ylabel('z [m]')
ax2.set_zlabel('[V]')
fig2.colorbar(surf, shrink=0.4, aspect=20, pad=-0.025)
axes_width = fig2.get_size_inches()[1]*(fig2.subplotpars.right-fig2.subplotpars.left)
right =1.095
left =-.15
fig2.subplots_adjust(left=left,right=right) 
fig2.set_size_inches((fig2.get_size_inches()[0],axes_width/(right-left)))
ax2.tick_params(axis='both', pad=15)
axes_height = fig2.get_size_inches()[1]*(fig2.subplotpars.top-fig2.subplotpars.bottom)
top = 1.15
bottom=-.09
fig2.subplots_adjust(top=top,bottom=bottom)             
fig2.set_size_inches((fig2.get_size_inches()[0],axes_height/(top-bottom)))

## figure save

coord_dict


pdict['fig1_bool'] = True

if pdict['figsave_bool']:
    if pdict['figpt5_bool']:
        figpt5.savefig(fig_exp_dir + 'potxsec_compare.pdf', dpi=300, format='pdf')  ## Comparison to analytical solution
    if pdict['fig1_bool']:
        fig1.savefig(fig_exp_dir + 'assembly4_sim.pdf', dpi=300, format='pdf')
    if pdict['fig1pt5_bool']:
        fig1pt5.savefig(fig_exp_dir + 'fieldxsec_sim.pdf', dpi=300, format='pdf')
    if pdict['fig2_bool']:
        fig2.savefig(fig_exp_dir + 'fieldmap_sim.pdf', dpi=300, format='pdf')

## data save
if pdict['potsave_bool']:
    np.savez('outfile', rho.reshape(N,N), z.reshape(N,N), V.reshape(N,N))