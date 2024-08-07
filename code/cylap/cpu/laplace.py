import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import lil_matrix
from scipy.sparse import spdiags


# Computes Laplace's equation in cartesian and cylindrical coordinates
# For some related detailed documentation: Numerical recipies (3rd edition) (Chapter 20 [Partial Differential Equations])


## Initialize fields

def init_coords(pdict):
    """ 
    Looks at params file to start implementing coordinate choices for simulation
    """
    if pdict['coords'] == 'cylindrical':
        
        i_rho = np.arange(pdict['origin'][0],pdict['N'][0])
        i_z  = np.arange(pdict['origin'][1],pdict['N'][1])                                          
        rho_ = i_rho*pdict['res'][0]
        z_ = i_z*pdict['res'][1]
        rho = (rho_ * np.ones((pdict['N'][0],1)))
        rho = rho.reshape(pdict['N'][0]*pdict['N'][1],1)
        z  = (np.ones((pdict['N'][1],1)) * z_).T
        z = z.reshape(pdict['N'][0]*pdict['N'][1],1)
        invrho = 1/rho
        invrho[rho==0] = 0                                   # addresses inf elements
        
        coord_dict = {
            'coords' : {
                'rho' : np.round(rho, decimals=pdict['res_exp'][0]),
                'z' : np.round(z, decimals=pdict['res_exp'][1]),
                'invrho' : np.round(invrho, decimals=pdict['res_exp'][0])},
            'indices' : {
                'rho' : i_rho,
                'z' : i_z}
            }
        
    #elif pdict['coords'] == 'cartesian': 
    
    return coord_dict
    

def indx(icoord1, icoord2, N):
    """
    formalized lambda function reshaping potential (vectorizing V):
    indx = lambda i_rho, i_z : np.int32(i_rho + i_z*(N))
    """
    return np.int32(icoord1 + icoord2*N)


def idx_match(vec,N,step):
    """
    Acquire nearest matching ind(ex/ices) for queried location(s) in potential map
    """
    idx = np.int32(np.round(vec/step, decimals=0))
    idx = 1 if idx<1 else N if idx>N else idx
    return idx


def init_V(N):
    """
    Initialize (square) potential map
    """
    return np.zeros((N**2,1))
    

def build_lambd(i1, i2, N):
    """
    Constructs a matrix for a lambda function, which operates on all available indices in the simulation.
    Preallocates memory so that the indx function doesn't need to be used twice (reducing computations).
    """
    LAMBD = np.array([indx(i, i2, N) for i in i1])
    return LAMBD
        
    

def bc_set(pdict, BC, N, V):
    """
    Establishes simulation boundary conditions
    """
    #global R, d, step, idx, V, rho, z, bc0set
    
    #Plate bcs
    if pdict['coords'] == 'cylindrical':
        
        #Setting up the edge boundaries (for faster convergence)  
        if not bc0set:
            rho0 = False
            rhoend = np.interp(np.arange(0,N), np.array([0,N-1]),np.array([pdict['back_plate']['voltage'],pdict['front_plate']['voltage']])).reshape(N,1)
            z0 = pdict['back_plate']['voltage']
            zend = pdict['front_plate']['voltage']
            edge_vals =np.array(rho0, rhoend, z0, zend)
            V = bc_edge(pdict, edge_vals, V)
            bc0set = True
            
            
        #Set potentials
        for i in range(BC['cont']):
            V = set_pot(V,BC[i]['coords'],BC[i]['values'],LAMBD)
        
        # exponential boundary conditions
        V0 = 0
        R0 = 1
        V[idx(np.arange(0,N),N-1)] = V0  + np.exp(-step/R0)*(V[idx(np.arange(0,N), N-2)]-V0)
        V[idx(np.arange(0,N),0)] = V0 + np.exp(-step/R0)*(V[idx(np.arange(0,N),1)]-V0)
        V[idx(N-1,np.arange(0,N))]= V0 + np.exp(-step/R0)*(V[idx(N-2, np.arange(0,N))]-V0) 
                 
    return V
                 

                 

# Constructing the operator(s)
    
def build_lap(pdict, LAMBD, i_rho):
    """
    constructs first order structure of the laplace operator
    """
    if pdict['coords'] == 'cylindrical':
        op_shape = (pdict['N'][0]**2, pdict['N'][1]**2)
        lap = lil_matrix(op_shape,dtype=pdict['bitres'])
        lap[LAMBD[0,1:-1], LAMBD[0,1:-1]] = -6
        lap[LAMBD[0,1:-1], LAMBD[1,1:-1]] = 4
        lap[LAMBD[0,1:-1], LAMBD[0,:-2]] = 1
        lap[LAMBD[0,1:-1], LAMBD[0,2:]] = 1
        lap[LAMBD[1:-1,1:-1], LAMBD[1:-1,1:-1]] = -4
        lap[LAMBD[1:-1,1:-1], LAMBD[1:-1,:-2]] = 1
        lap[LAMBD[1:-1,1:-1], LAMBD[1:-1,2:]] = 1
        lap[LAMBD[1:-1,1:-1], LAMBD[:-2,1:-1]]= 1 - ((1/2)/(i_rho[1:-1]))
        lap[LAMBD[1:-1,1:-1], LAMBD[2:,1:-1]]= 1 + ((1/2)/(i_rho[1:-1]))
        lap = lap/(pdict['res'][0]**2)
   # if block_bool == True:
   #     lap[LAMBD[0,0], LAMBD[0,0]] = -6
   #     lap[LAMBD[0,0], LAMBD[1,0]] = 4
   #     lap[LAMBD[1:-1,0], LAMBD[1:-1,0]]=-4
   #     lap[LAMBD[1:-1,0], LAMBD[1:-1,1]] = 
        #if pdict['torch'] == True:
        #lap.
        #lap.to_sparse(layout=torch.sparse_coo)
            
           ## idx_1 = LAMBD[0,1:-1]
           ## idx_2 = LAMBD[1,1:-1]
           ## idx_3 = LAMBD[0,:-2]
           ## idx_4 = LAMBD[0,2:]
           ## size_ = (pdict['N'][0]-2)**2
           ## idx_5 = LAMBD[1:-1,1:-1].reshape(size_)
           ## idx_6 = LAMBD[1:-1,:-2].reshape(size_)
           ## idx_7 = LAMBD[1:-1,2:].reshape(size_)
           ## idx_8 = LAMBD[:-2,1:-1].reshape(size_)
           ## idx_9 = LAMBD[2:,1:-1].reshape(size_)
           ## ones_1 = np.ones(idx_1.shape)
           ## ones_2 = np.ones(idx_5.shape)
           ## const_ = (np.ones((1,i_rho[1:-1].shape[0])).T*(((1/2)/(i_rho[1:-1])))).reshape(size_)
           ## lap1 = torch.sparse_coo_tensor(np.array([idx_1, idx_1]), -6*ones_1, op_shape, dtype=torch.float32)
           ## lap2 = torch.sparse_coo_tensor(np.array([idx_1, idx_2]), 4*ones_1, op_shape, dtype=torch.float32)
           ## lap3 = torch.sparse_coo_tensor(np.array([idx_1, idx_3]), ones_1, op_shape, dtype=torch.float32)
           ## lap4 = torch.sparse_coo_tensor(np.array([idx_1, idx_4]), ones_1, op_shape, dtype=torch.float32)
           ## lap5 = torch.sparse_coo_tensor(np.array([idx_5, idx_5]), -4*ones_2, op_shape, dtype=torch.float32)
           ## lap6 = torch.sparse_coo_tensor(np.array([idx_5, idx_6]), ones_2, op_shape, dtype=torch.float32)
           ## lap7 = torch.sparse_coo_tensor(np.array([idx_5, idx_7]), ones_2, op_shape, dtype=torch.float32)
           ## lap8 = torch.sparse_coo_tensor(np.array([idx_5, idx_8]), 1 - const_, op_shape, dtype=torch.float32)
           ## lap9 = torch.sparse_coo_tensor(np.array([idx_5, idx_9]), 1 + const_, op_shape, dtype=torch.float32)
           ## lap_ = lap1 + lap2 + lap3 + lap4 + lap5 + lap6 + lap7 + lap8 + lap9
           ## lap = lap_/(pdict['res'][0]**2)

    #elif pdict['coords'] == 'cartesian':
        
    return lap

def build_grad(pdict, LAMBD):
    """
    Gradient operators 
    """
    if pdict['coords'] == 'cylindrical': 
        if pdict['torch'] == True:
            
            idx1 = LAMBD[1:-1,:]
            idx2 = LAMBD[:-2,:]
            idx3 = LAMBD[2:,:]
            idx4 = LAMBD[:,1:-1]
            idx5 = LAMBD[:,:-2]
            idx6 = LAMBD[:,2:]
            
            gradrho1 = torch.sparse_coo_tensor(np.array([idx1, idx2]), -1/2, op_shape, dtype=pdict['bitres'])
            gradrho2 = torch.sparse_coo_tensor(np.array([idx1, idx3]), -1/2, op_shape, dtype=pdict['bitres'])
            GRADrho = (gradrho1+gradrho2)/pdict['res'][0]
            
            gradrhopos1 = torch.sparse_coo_tensor(np.array([idx1, idx1]), -1, op_shape, dtype=pdict['bitres'])
            gradrhopos2 = torch.sparse_coo_tensor(np.array([idx1, idx3]), -1, op_shape, dtype=pdict['bitres'])
            GRADrhopos = (gradrhopos1+gradrhopos2)/pdict['res'][0]
            
            gradrhoneg1 = torch.sparse_coo_tensor(np.array([idx1, idx2]), -1, op_shape, dtype=pdict['bitres'])
            gradrhoneg2 = torch.sparse_coo_tensor(np.array([idx1, idx1]), -1, op_shape, dtype=pdict['bitres'])
            GRADrhoneg = (gradrhoneg1+gradrhoneg2)/pdict['res'][0]
            
            gradz1 = torch.sparse_coo_tensor(np.array([idx4, idx5]), -1/2, op_shape, dtype=pdict['bitres'])
            gradz2 = torch.sparse_coo_tensor(np.array([idx4, idx6]), -1/2, op_shape, dtype=pdict['bitres'])
            GRADz = (gradz1+gradz2)/pdict['res'][1]
            
            gradzpos1 = torch.sparse_coo_tensor(np.array([idx4, idx4]), -1, op_shape, dtype=pdict['bitres'])
            gradzpos2 = torch.sparse_coo_tensor(np.array([idx4, idx6]), 1, op_shape, dtype=pdict['bitres'])
            GRADzpos = (gradzpos1 + gradzpos2)/pdict['res'][1]
            
            gradzpos1 = torch.sparse_coo_tensor(np.array([idx4, idx5]), -1, op_shape, dtype=pdict['bitres'])
            gradzpos2 = torch.sparse_coo_tensor(np.array([idx4, idx4]), 1, op_shape, dtype=pdict['bitres'])
            GRADzneg = (gradzpos1 + gradzpos2)/pdict['res'][1]
            
        else:
            
            init_sparmat = lambda shape, res : lil_matrix(shape, dtype = res)
            
            op_shape = (pdict['N'][0]**2, pdict['N'][1]**2)
            
            GRADrho = init_sparmat(op_shape, pdict['bitres'])
            GRADrho[LAMBD[1:-1,:], LAMBD[:-2,:]] = -1/2
            GRADrho[LAMBD[1:-1,:], LAMBD[2:,:]]= 1/2
            GRADrho[LAMBD[0,1:-1],LAMBD[0,1:-1]]=-2/pdict['res'][0]
            GRADrho[LAMBD[0,1:-1],LAMBD[1,1:-1]]= 2/pdict['res'][0]
            GRADrho = GRADrho/pdict['res'][0]

            GRADrhopos = init_sparmat(op_shape, pdict['bitres'])
            GRADrhopos[LAMBD[1:-1,:], LAMBD[1:-1,:]] = -1
            GRADrhopos[LAMBD[1:-1,:], LAMBD[2:,:]]= 1
            GRADrhopos = GRADrhopos/pdict['res'][0]

            GRADrhoneg = init_sparmat(op_shape, pdict['bitres'])
            GRADrhoneg[LAMBD[1:-1,:], LAMBD[:-2,:]] = -1
            GRADrhoneg[LAMBD[1:-1,:], LAMBD[1:-1,:]]= 1
            GRADrhoneg = GRADrhoneg/pdict['res'][0]

            GRADz= init_sparmat(op_shape, pdict['bitres'])
            GRADz[LAMBD[:,1:-1], LAMBD[:,:-2]] = -1/2
            GRADz[LAMBD[:,1:-1], LAMBD[:,2:]]= 1/2
            GRADz = GRADz/pdict['res'][1]

            GRADzpos= init_sparmat(op_shape, pdict['bitres'])
            GRADzpos[LAMBD[:,1:-1], LAMBD[:,1:-1]] = -1
            GRADzpos[LAMBD[:,1:-1], LAMBD[:,2:]]= 1
            GRADzpos = GRADzpos/pdict['res'][1]

            GRADzneg= init_sparmat(op_shape, pdict['bitres'])
            GRADzneg[LAMBD[:,1:-1], LAMBD[:,:-2]] = -1
            GRADzneg[LAMBD[:,1:-1], LAMBD[:,1:-1]]= 1
            GRADzneg = GRADzneg/pdict['res'][1]
        
        
    return GRADrho, GRADrhopos, GRADrhoneg, GRADz, GRADzpos, GRADzneg


def build_disp(pdict, LAMBD):
    """
    Displacement operators
    """
    
    if pdict['coords'] == 'cylindrical':
        DISPrhopos = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPrhopos[LAMBD[1:,:], LAMBD[:-1,:]] = 1

        DISPrhoneg = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPrhoneg[LAMBD[:-1,:], LAMBD[1:,:]] = 1

        DISPzpos = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPzpos[LAMBD[:,1:], LAMBD[:,:-1]] = 1

        DISPzneg = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPzneg[LAMBD[:,:-1], LAMBD[:,1:]] = 1
        
    return DISPrhopos, DISPrhoneg, DISPzpos, DISPzneg
             
def build_LAP(pdict, coord_dict, lap, grad, disp, chi_e):
    """
    full laplace operator (dielectric considerations)
    """
    if pdict['coords'] == 'cylindrical':
        GRADrho = grad[0]
        GRADrhopos = grad[1]
        GRADrhoneg = grad[2]
        GRADz = grad[3]
        GRADzpos = grad[4]
        GRADzneg = grad[5]
        
        DISPrhopos = disp[0]
        DISPrhoneg = disp[1]
        DISPzpos = disp[2]
        DISPzneg = disp[3]
    
        chi_e_half = chi_e/2

        CHI1 = spdiags((1/(1+chi_e_half)).T,0, pdict['N'][0]*pdict['N'][1], pdict['N'][0]*pdict['N'][1], format='lil')
        CHI2 = spdiags((chi_e_half*coord_dict['coords']['invrho']).T,0, pdict['N'][0]*pdict['N'][1], pdict['N'][0]*pdict['N'][1], format='lil')
        DNEG = spdiags(DISPrhoneg.dot(chi_e_half).T,0, pdict['N'][0]*pdict['N'][1], pdict['N'][0]*pdict['N'][1], format='lil')
        DPOS = spdiags(DISPrhopos.dot(chi_e_half).T,0, pdict['N'][0]*pdict['N'][1], pdict['N'][0]*pdict['N'][1], format='lil')
        ZNEG = spdiags(DISPzneg.dot(chi_e_half).T,0, pdict['N'][0]*pdict['N'][1], pdict['N'][0]*pdict['N'][1], format='lil')
        ZPOS = spdiags(DISPzpos.dot(chi_e_half).T,0, pdict['N'][0]*pdict['N'][1], pdict['N'][0]*pdict['N'][1], format='lil')
        LAP = lap + CHI1.dot(CHI2.dot(GRADrho) + (DNEG.dot(GRADrhopos) - DPOS.dot(GRADrhoneg))/pdict['res'][0] + (ZNEG.dot(GRADzpos) - ZPOS.dot(GRADzneg))/pdict['res'][1]) 
        
    #elif pdict['coords'] == 'cartesian':
        
    return LAP



def anal_sol(pdict):
    z_p1 = pdict['front_plate']['zpos']
    z_p2 = pdict['back_plate']['zpos']
    d_plates = z_p1 - z_p2
    V_p1 = pdict['front_plate']['voltage']
    V_p2 = pdict['back_plate']['voltage']
    V_diff = V_p1 - V_p2
    d_opt = pdict['optic']['thickness']
    d_sub = pdict['optic']['sub_thickness']
    d_coat = pdict['optic']['coat_thickness']
    d_air = pdict['cap_params']['d_air']
    z_opt = pdict['optic']['z_com']
    p1_2_opt  = z_p1 - (d_opt/2.0) - z_opt
    opt_2_p2 = z_opt - (d_opt/2.0) - z_p2
    eps_air = pdict['cap_params']['air_eps']
    eps_sub = pdict['optic']['sub_eps']   
    eps_coat = pdict['optic']['coat_eps']
    CoA = pdict['cap_params']['cap_div_area']
    cap_ratio = CoA
    air_ratio = d_air/eps_air
    sub_ratio = d_sub/eps_sub
    coat_ratio = d_coat/eps_coat
    V_air = cap_ratio*air_ratio*V_diff
    V_coat = cap_ratio*coat_ratio*V_diff
    V_sub = cap_ratio*sub_ratio*V_diff
    

    #E_front = V_diff/(p1_2_opt + opt_2_p2 + (d_opt-d_coat)/eps_sub + d_coat/eps_coat)
    #E_sub = V_diff/((p1_2_opt + opt_2_p2 + d_coat/eps_coat)*eps_sub + (d_opt-d_coat))
    #E_coat = V_diff/( (p1_2_opt + opt_2_p2 + (d_opt-(d_coat))/eps_sub)*eps_coat + d_coat)
    #E_back = E_front
    z_anal = z_p2+np.array([0, opt_2_p2, (opt_2_p2 + d_sub), (opt_2_p2 + d_sub + d_coat), (opt_2_p2 + d_sub + d_coat + opt_2_p2)])
    V_anal = np.array([V_p2, V_p2 + V_air, V_p2 + V_air + V_sub, V_p2 + V_air + V_sub + V_coat, V_p1])
    anal_dict = {
        'V_anal' : lambda z: np.interp(z, z_anal, V_anal)
        }
    return anal_dict


def pltxsect(loc_params, coord_dict, V): 
    if loc_params['cross_section_coord'] == 'z':
        rho_ = coord_dict['coords']['rho'] == np.around(loc_params['rho'], int(abs(np.log10(coord_dict['coords']['rho'][1]))))
        z_ = np.logical_and(coord_dict['coords']['z']<=loc_params['z1_bound'], coord_dict['coords']['z']>=loc_params['z2_bound'])
        plt.plot(coord_dict['coords']['z'][np.logical_and(rho_,z_)],V[np.logical_and(rho_, z_)])
    if loc_params['cross_section_coord'] == 'rho':
        z_ = coord_dict['coords']['z'] == np.around(loc_params['z'], int(abs(np.log10(coord_dict['coords']['z'][1]))))
        rho_ = np.logical_and(coord_dict['coords']['rho']<=loc_params['rho2_bound'], coord_dict['coords']['rho']>=loc_params['rho1_bound'])
        plt.plot(coord_dict['coords']['rho'][np.logical_and(rho_,z_)],V[np.logical_and(rho_, z_)])

def grabxsect(loc_params, coord_dict, res_exp, V): 
    if loc_params['cross_section_coord'] == 'z':
        rho_ = coord_dict['coords']['rho'] == np.around(loc_params['rho'], res_exp[1])
        z_ = np.logical_and(coord_dict['coords']['z']<=loc_params['z1_bound'], coord_dict['coords']['z']>=loc_params['z2_bound'])
        xcoord = coord_dict['coords']['z'][np.logical_and(rho_,z_)]
        V_xsec = V[np.logical_and(rho_, z_)]
    if loc_params['cross_section_coord'] == 'rho':
        z_ = coord_dict['coords']['z'] == np.around(loc_params['z'], res_exp[1])
        rho_ = np.logical_and(coord_dict['coords']['rho']<=loc_params['rho2_bound'], coord_dict['coords']['rho']>=loc_params['rho1_bound'])
        xcoord = coord_dict['coords']['rho'][np.logical_and(rho_,z_)]
        V_xsec = V[np.logical_and(rho_, z_)]
    return xcoord, V_xsec


