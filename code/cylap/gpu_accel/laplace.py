import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import lil_matrix
from scipy.sparse import spdiags
import torch
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_gpu 

# Computes Laplace's equation in cartesian and cylindrical coordinates
# For some related detailed documentation: Numerical recipies (3rd edition) (Chapter 20 [Partial Differential Equations])

## Solving Laplace's equation in cylindrical coordinates

#class laplace:
#    import set_params
#    pdict = set_params.pdict
#    def __init__(self, pdict):
#        self

## Initialize fields

def params():
    import set_params
    return set_params.pdict

pdict = params()

def dp2mg(bl_coord, tr_coord, res_exp, offset):
    roun = lambda val, dec : np.round(int(val)*(10.0**-dec), decimals=dec)
    c1_range = np.arange(bl_coord[0], tr_coord[0], res_exp[0]) + offset[0]
    c2_range = np.arange(bl_coord[1], tr_coord[1], res_exp[1]) + offset[1]
    c1_, c2_ = np.meshgrid(c1_, c2_, indexing='xy')
    return c1_, c2_

def expand_mg(c1_mesh, c2_mesh, expand_val, res, bool_edges = [True, True, True, True]):
    offset = [c1_mesh.min(), c2_mesh.min()]
    max_vals = [c1_mesh.max(), c2_mesh.max()]
    if bool_edges[0]:
        offset[0] = -1*expand_val + offset[0] 
    if bool_edges[1]:
        offset[1] = -1*expand_val + offset[1]
    if bool_edges[2]:
        max_vals[0] = expand_val + max_vals[0]
    if bool_edges[3]:
        max_vals[1] = expand_val + max_vals[1]
    c1_range = np.arange(0, max_vals[0]-offset[0], res[0] ) + offset[0]
    c2_range = np.arange(0, max_vals[1]-offset[1], res[1]) + offset[1]
    c1_out, c2_out = np.meshgrid(c1_range, c2_range, indexing='xy')
    return np.round(c1_out, decimals = abs(int(np.log10(res[0])))), np.round(c2_out, decimals=abs(int(np.log10(res[1]))))
    


def init_coords(pdict):
    """ 
    Looks at params file to start implementing coordinate choices for simulation
    """
    if pdict['coords'] == 'cylindrical':
        if pdict['gpu_accel'] and pdict['torch']:
            i_rho = torch.arange(pdict['origin'][0],pdict['N'][0])
            i_z = torch.arange(pdict['origin'][1],pdict['N'][1])
            rho_ = i_rho*torch.tensor(pdict['res'][0])
            z_ = i_z*torch.tensor(pdict['res'][1])
            rho, z = torch.meshgrid(rho_, z_, indexing='xy')
            invrho = 1/rho
            invrho[rho==0] = 0
        else:
            i_rho = np.arange(pdict['origin'][0],pdict['N'][0])
            i_z  = np.arange(pdict['origin'][1],pdict['N'][1])   
            rho_ = i_rho*pdict['res'][0]
            z_ = i_z*pdict['res'][1]
            z, rho = np.meshgrid(rho_, z_, indexing='xy')
            invrho = 1/rho
            invrho[rho==0] = 0                                   # addresses inf elements
            
        coord_dict = {
            'coords' : {
                'rho' : np.round(rho,pdict['res_exp'][0]),
                'z' : np.round(z,pdict['res_exp'][1]),
                'invrho' : np.round(invrho, pdict['res_exp'][0])}, #np.round(invrho,abs(int(np.log10(pdict['res'][0]))))},
            'indices' : {
                'rho' : i_rho,
                'z' : i_z}
            }
        rel_const = pdict['res']/pdict['roi_res']
        roun = lambda val, dec : np.round(int(val)*(10.0**-dec), decimals=dec)
        if pdict['roi_mult'] != 0:
            pdict['roi_offset'] = [0, np.round(pdict['loc_params']['front of optic']['z'] - int((pdict['N'][0]-1)/2)*(10.0**-pdict['roi_res_exp'][0]), decimals= pdict['roi_res_exp'][0])]
            roirho_ = np.arange(0, roun(pdict['N'][0], pdict['roi_res_exp'][0]), pdict['res'][0]*pdict['roi_mult']) + pdict['roi_offset'][0]
            roiz_ = np.arange(0, roun(pdict['N'][1], pdict['roi_res_exp'][1]), pdict['res'][1]*pdict['roi_mult']) + pdict['roi_offset'][1]
            roiz, roirho = np.meshgrid(roiz_, roirho_, indexing='xy')
            coord_dict['roi'] = {
                    'rho' : roirho, 
                    'z' : roiz}
            coord_dict['roi']['invrho'] = 1/coord_dict['roi']['rho']
            coord_dict['roi']['invrho'][coord_dict['roi']['rho'] == 0] = 0 
   
    #elif pdict['coords'] == 'cartesian': 
    
    return coord_dict

def gpu_check(gpu_bool = True):
    mps_gpu_bool = torch.backends.mps.is_available()
    nvid_gpu_bool = torch.cuda.is_available()
    if (mps_gpu_bool or nvid_gpu_bool) and gpu_bool:
        
        if mps_gpu_bool:
            device = 'mps'
            print("GPU acceleration via mps is available")
            return mps_gpu_bool, device
        elif nvid_gpu_bool:
            device = 'cuda'
            torch.cuda.empty_cache()
            torch.cuda.mem_get_info()
            torch.cuda.synchronize()
            print("GPU acceleration via cuda is available")
            return nvid_gpu_bool, device
        
    else:
        print("GPU acceleration via metal framework and cuda is not available")
        device = 'cpu'
        return False, device

def vectorize2d(tensor_2d):
    if str(tensor_2d.dtype)[:5] == 'torch':
        size = tensor_2d.numel()
    else:
        size = tensor_2d.size
    vec = tensor_2d.reshape(size,1)
    return vec

def indx(icoord1, icoord2, pdict):
    """
    formalized lambda function reshaping potential (vectorizing V):
    indx = lambda i_rho, i_z : np.int32(i_rho + i_z*(N))
    """
    if pdict['gpu_accel'] and pdict['torch']:
        indxs = (icoord1 + icoord2*pdict['N'][0]).type(torch.int)
    else:
        indxs = (icoord1 + icoord2*pdict['N'][0]).astype(int)
    return indxs

def init_V(pdict, coord_dict, vec_ind=True):
    """
    Initialize (square) potential map (vectorized)
    Option for a coordinates of the vectorized version can be toggled with vec_ind tag.
    """
    if (pdict['gpu_accel'] and pdict['torch']):
        V = torch.zeros(pdict['vec_shape']).reshape(pdict['vec_shape'])
        if vec_ind:
            irho , iz = torch.meshgrid(coord_dict['indices']['rho'], coord_dict['indices']['z'], indexing='ij')
    else:
        V = np.zeros(pdict['vec_shape']).reshape(pdict['vec_shape'])
        if vec_ind:
            irho, iz = np.meshgrid(coord_dict['indices']['rho'], coord_dict['indices']['z'], indexing='xy')
        
    if vec_ind :
        LAMBD = indx(irho, iz, pdict)
        return V, LAMBD
    else: 
        return V
    
def bc_set(pdict, BC, N, V):
    """
    Establishes simulation (Cauchy boundary conditions)
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
        R0 = 1.0
        V[idx(np.arange(0,N),N-1)] = V0  + np.exp(-step/R0)*(V[idx(np.arange(0,N), N-2)]-V0)
        V[idx(np.arange(0,N),0)] = V0 + np.exp(-step/R0)*(V[idx(np.arange(0,N),1)]-V0)
        V[idx(N-1,np.arange(0,N))]= V0 + np.exp(-step/R0)*(V[idx(N-2, np.arange(0,N))]-V0) 
                 
    return V
                 
## CONSTRUCTING THE OPERATORS ##
    
def build_lap(pdict, LAMBD, i_rho, dx_dy=pdict['res']):
    """
    constructs first order structure of the laplace operator
    """
    if pdict['coords'] == 'cylindrical':
        op_shape = (pdict['N'].prod(), pdict['N'].prod())
        idx_1 = LAMBD[0,1:-1]
        idx_2 = LAMBD[1,1:-1]
        idx_3 = LAMBD[0,:-2]
        idx_4 = LAMBD[0,2:]
        idx_5 = LAMBD[1:-1,1:-1]
        idx_6 = LAMBD[1:-1,:-2]
        idx_7 = LAMBD[1:-1,2:]
        idx_8 = LAMBD[:-2,1:-1]
        idx_9 = LAMBD[2:,1:-1]
                      
        if pdict['gpu_accel'] and pdict['torch']:
            size_ = ((pdict['N'][0]-2)**2,1)
            idx_5 = idx_5.flatten()
            idx_6 = idx_6.flatten()
            idx_7 = idx_7.flatten()
            idx_8 = idx_8.flatten()
            idx_9 = idx_9.flatten()
            ones_1 = torch.ones(idx_1.numel())
            ones_2 = torch.ones(idx_5.numel()).flatten()
            const_ = (torch.ones(1, i_rho[1:-1].numel())*(.5/(i_rho[1:-1].reshape(i_rho[1:-1].numel(),1)))).flatten()
            idx_i = torch.cat((idx_1, idx_1, idx_1, idx_1, idx_5, idx_5, idx_5, idx_5, idx_5))
            idx_j = torch.cat((idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7, idx_8, idx_9))
            lap_vals = torch.cat((-6*ones_1, 4*ones_1, ones_1, ones_1, -4*ones_2, ones_2, ones_2, (1-const_), (1+const_)))
            #lap_coords = torch.tensor(np.array([idx_i, idx_j]))
            idxs_ = torch.cat((idx_i, idx_j)).reshape(2,(idx_1.numel()*4 + idx_5.numel()*5))
            lap = torch.sparse_coo_tensor(idxs_, lap_vals/(dx_dy[0]**2), size=op_shape, dtype=torch.float32)
        else:
            lap = lil_matrix(op_shape,dtype=pdict['bitres'])
            lap[idx_1, idx_1] = -6
            lap[idx_1, idx_2] = 4
            lap[idx_1, idx_3] = 1
            lap[idx_1, idx_4] = 1
            lap[idx_5, idx_5] = -4
            lap[idx_5, idx_6] = 1
            lap[idx_5, idx_7] = 1
            lap[idx_5, idx_8]= 1 - ((1/2)/(i_rho[1:-1]))
            lap[idx_5, idx_9]= 1 + ((1/2)/(i_rho[1:-1]))
            lap = (lap/(dx_dy[0]**2)).tocsr(copy=False)
        
    #elif pdict['coords'] == 'cartesian':
        
    return lap

def build_grad(pdict, LAMBD, dx_dy=pdict['res']):
    """
    Gradient operators 
    """
    if pdict['coords'] == 'cylindrical':
        idx1 = vectorize2d(LAMBD[1:-1,:])
        idx2 = vectorize2d(LAMBD[:-2,:])
        idx3 = vectorize2d(LAMBD[2:,:])
        idx4 = vectorize2d(LAMBD[:,1:-1])
        idx5 = vectorize2d(LAMBD[:,:-2])
        idx6 = vectorize2d(LAMBD[:,2:])
        op_shape = (pdict['N'].prod(), pdict['N'].prod())

        if pdict['gpu_accel'] and pdict['torch']:
            
                idx1sz_dbl = (2, idx1.numel()*2)
                idx4sz_dbl = (2, idx4.numel()*2)
            
                GRADrho = torch.sparse_coo_tensor(torch.cat((idx1,idx1,idx2,idx3)).reshape(idx1sz_dbl), torch.cat((-1*torch.ones(idx1.numel()), torch.ones(idx1.numel())))*(.5/dx_dy[0]), size=op_shape, dtype=(pdict['bitres']))
            
                GRADrhopos = torch.sparse_coo_tensor(torch.cat((idx1, idx1, idx1, idx3)).reshape(idx1sz_dbl), torch.cat((-1*torch.ones(idx1.numel()), torch.ones(idx1.numel())))/dx_dy[0], size=op_shape, dtype=(pdict['bitres']))
            
                GRADrhoneg = torch.sparse_coo_tensor(torch.cat((idx1, idx1, idx2,idx1)).reshape(idx1sz_dbl), torch.cat((-1*torch.ones(idx1.numel()), torch.ones(idx1.numel())))/dx_dy[0], size=op_shape, dtype=(pdict['bitres']))

                GRADz = torch.sparse_coo_tensor(torch.cat((idx4, idx4, idx5,idx6)).reshape(idx4sz_dbl), torch.cat((-1*torch.ones(idx4.numel()),torch.ones(idx4.numel())))*(.5/dx_dy[1]), size=op_shape, dtype=(pdict['bitres']))
            
                GRADzpos = torch.sparse_coo_tensor(torch.cat((idx4, idx4, idx4,idx6)).reshape(idx4sz_dbl), torch.cat((-1*torch.ones(idx4.numel()),torch.ones(idx4.numel())))/dx_dy[1], size=op_shape, dtype=(pdict['bitres']))
           
                GRADzneg = torch.sparse_coo_tensor(torch.cat((idx4, idx4, idx5, idx4)).reshape(idx4sz_dbl), torch.cat((-1*torch.ones(idx4.numel()),torch.ones(idx4.numel())))/dx_dy[1], size=op_shape, dtype=(pdict['bitres']))
           
        else:
            
            idx1sz_dbl = (2, idx1.shape[0]*2)
            idx4sz_dbl = (2, idx4.shape[0]*2)
            
            init_sparmat = lambda shape, res : lil_matrix(shape, dtype = res)
            
            op_shape = (pdict['N'].prod(), pdict['N'].prod())
            
            GRADrho = init_sparmat(op_shape, pdict['bitres'])
            GRADrho[idx1,idx2] = -1/2
            GRADrho[idx1,idx3]= 1/2
            GRADrho = (GRADrho/dx_dy[0]).tocsr(copy=False)

            GRADrhopos = init_sparmat(op_shape, pdict['bitres'])
            GRADrhopos[idx1,idx1] = -1
            GRADrhopos[idx1,idx3]= 1
            GRADrhopos = (GRADrhopos/dx_dy[0]).tocsr(copy=False)

            GRADrhoneg = init_sparmat(op_shape, pdict['bitres'])
            GRADrhoneg[idx1,idx2] = -1
            GRADrhoneg[idx1,idx1]= 1
            GRADrhoneg = (GRADrhoneg/dx_dy[0]).tocsr(copy=False)

            GRADz= init_sparmat(op_shape, pdict['bitres'])
            GRADz[idx4,idx5] = -1/2
            GRADz[idx4,idx6]= 1/2
            GRADz = (GRADz/dx_dy[1]).tocsr(copy=False)

            GRADzpos= init_sparmat(op_shape, pdict['bitres'])
            GRADzpos[idx4,idx4] = -1
            GRADzpos[idx4,idx6]= 1
            GRADzpos = (GRADzpos/dx_dy[1]).tocsr(copy=False)

            GRADzneg= init_sparmat(op_shape, pdict['bitres'])
            GRADzneg[idx4,idx5] = -1
            GRADzneg[idx4,idx4]= 1
            GRADzneg = (GRADzneg/dx_dy[1]).tocsr(copy=False)
        
        
    return GRADrho, GRADrhopos, GRADrhoneg, GRADz, GRADzpos, GRADzneg


def build_disp(pdict, LAMBD, dx_dy=pdict['res']):
    """
    Displacement operators
    """ 
    idx1 = vectorize2d(LAMBD[1:,:])
    idx2 = vectorize2d(LAMBD[:-1,:])
    idx3 = vectorize2d(LAMBD[:,1:])
    idx4 = vectorize2d(LAMBD[:,:-1])
    
    op_shape = (pdict['N'][0]**2, pdict['N'][1]**2)
    
    if pdict['gpu_accel'] and pdict['torch']:   
        
        idx_shape = (2, idx1.numel())
        
        DISPrhopos = torch.sparse_coo_tensor(torch.cat((idx1, idx2)).reshape(idx_shape), torch.ones(idx1.numel()), op_shape, dtype=(pdict['bitres']))
        DISPrhoneg = torch.sparse_coo_tensor(torch.cat((idx2, idx1)).reshape(idx_shape), torch.ones(idx1.numel()), op_shape, dtype=(pdict['bitres']))
        DISPzpos = torch.sparse_coo_tensor(torch.cat((idx3, idx4)).reshape(idx_shape), torch.ones(idx3.numel()), op_shape, dtype=(pdict['bitres']))
        DISPzneg =  torch.sparse_coo_tensor(torch.cat((idx4, idx3)).reshape(idx_shape), torch.ones(idx3.numel()), op_shape, dtype=(pdict['bitres']))
        
    else:
           
        idx_shape = (2, idx1.shape[0])
        
        DISPrhopos = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPrhopos[idx1, idx2] = 1

        DISPrhoneg = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPrhoneg[idx2, idx1] = 1

        DISPzpos = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPzpos[idx3, idx4] = 1

        DISPzneg = lil_matrix((pdict['N'][0]**2, pdict['N'][1]**2),dtype=pdict['bitres'])
        DISPzneg[idx4, idx3] = 1
        
    return DISPrhopos.tocsr(copy=False), DISPrhoneg.tocsr(copy=False), DISPzpos.tocsr(copy=False), DISPzneg.tocsr(copy=False)
             
def build_LAP(pdict, coord_dict_, lap, grad, disp, chi_e, dx_dy=pdict['res']):
    """
    full laplace operator (dielectric considerations)
    """
    if pdict['coords'] == 'cylindrical':
        shape_ = (pdict['N'].prod(),pdict['N'].prod())
        if pdict['gpu_accel'] and pdict['torch']:
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
            
            indx_diag = torch.arange(0, shape_[0]).repeat(2,1) 
            chi_e_half = (chi_e/2)

            nonzero_idxs = lambda x : torch.where(x != 0)[0]
            nonzero_vals = lambda y : y[nonzero_idxs(y)]

            def init_spar_tens(diag_tensor, shape_, dtype_):
                diagonals = nonzero_idxs(diag_tensor).repeat(2,1)
                values = diag_tensor[diagonals[0]]
                return torch.sparse_coo_tensor(diagonals, values, size=shape_, dtype=dtype_)

            CHI1 = init_spar_tens((1/(1+chi_e_half)).T[0], shape_, pdict['bitres'])
            CHI2 = init_spar_tens((chi_e_half*coord_dict_['invrho'].reshape(pdict['vec_shape'])).T[0], shape_, pdict['bitres'])
            DNEG = init_spar_tens((DISPrhoneg@chi_e_half).T[0], shape_, pdict['bitres'])
            DPOS = init_spar_tens((DISPrhopos@chi_e_half).T[0], shape_, pdict['bitres'])
            ZNEG = init_spar_tens((DISPzneg@chi_e_half).T[0], shape_, pdict['bitres'])
            ZPOS = init_spar_tens((DISPzpos@chi_e_half).T[0], shape_, pdict['bitres'])
            LAP = lap.add(torch.sparse.mm(CHI1,(torch.sparse.mm(CHI2,GRADrho).add((torch.sparse.mm(DNEG,GRADrhopos).sub(torch.sparse.mm(DPOS,GRADrhoneg)).mul(1/dx_dy[0]).coalesce()).add((torch.sparse.mm(ZNEG,GRADzpos).sub(torch.sparse.mm(ZPOS,GRADzneg))).mul(1/dx_dy[1]).coalesce())))))
            

            #Iteration considerations:
            res = torch.tensor(dx_dy[0]**2, dtype= torch.float32).to(device=pdict['device'])
            step = torch.tensor(pdict['iter_step'], dtype = torch.float32)
            LAP = LAP.coalesce().to_sparse_csr()
        else:
            N = pdict['N'].prod()
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

            CHI1 = spdiags((1/(1+chi_e_half)).T,0, N, N, format='lil')
            CHI2 = spdiags((chi_e_half*coord_dict_['invrho'].reshape(pdict['vec_shape'])).T,0, N, N, format='lil')
            DNEG = spdiags(DISPrhoneg.dot(chi_e_half).T,0, N, N, format='lil')
            DPOS = spdiags(DISPrhopos.dot(chi_e_half).T,0, N, N, format='lil')
            ZNEG = spdiags(DISPzneg.dot(chi_e_half).T,0, N, N, format='lil')
            ZPOS = spdiags(DISPzpos.dot(chi_e_half).T,0, N, N, format='lil')
            LAP = (lap + CHI1.dot(CHI2.dot(GRADrho) + (DNEG.dot(GRADrhopos) - DPOS.dot(GRADrhoneg))/dx_dy[0] + (ZNEG.dot(GRADzpos) - ZPOS.dot(GRADzneg))/dx_dy[1])).tocsr(copy=False)
            if pdict['gpu_accel']:
                LAP = csr_gpu(LAP)
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


