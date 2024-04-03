import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np
plt.style.use('../stylelib/surftex')

save_bool = True

export_dir = '../../../../figs/ALGAAS/'

file = 'a4out.npz'

import_dat = np.load(file)

list(import_dat.keys())

fig1 = plt.figure(figsize = (21,21))
ax = plt.axes(projection='3d') 
surf = ax.plot_surface(import_dat['arr_0'], import_dat['arr_1'], import_dat['arr_2'],rstride=1,cstride=1,cmap=cm.inferno,alpha=1,linewidth=10,rasterized=True)
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
ax.tick_params(axis='both', pad=12.5)
axes_height = fig1.get_size_inches()[1]*(fig1.subplotpars.top-fig1.subplotpars.bottom)
top = 1.15
bottom=-.09
fig1.subplots_adjust(top=top,bottom=bottom)             
fig1.set_size_inches((fig1.get_size_inches()[0],axes_height/(top-bottom)))
ax.set_box_aspect(None, zoom=0.91)

if save_bool:
    fig1.savefig(export_dir + 'assembly4_sim.pdf', dpi=300, format='pdf')
    



