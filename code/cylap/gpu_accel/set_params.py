import numpy as np
## Setting parameters
pdict ={
    'coords' : 'cylindrical'             ,                        # coordinate system chosen for simulation box
    'assembly' : 1                       ,                        # Establish plate geometry / location and voltage based on assembly 
    'origin' : np.array([0,0])           ,                        # Origin of the simulation space / map
    'size' : np.array([1, 1])*4e-2       ,                        # Size of simulation box [m] 
    'grid_res' : 1e-2                    ,                        # base 10 resolution 
    'roi_mult' : 1e-2                    ,                        # ROI multiplier (also base 10)
    'roi_pad'  : 1e-2                    ,                        # Padding around roi area for succesful griddata operation
    'iters' : int(1e5)                   ,                        # total number of time iterations
    'iter_step' : .01                    ,                        # time step
    'expbc' : False                      ,                        # Exponential boundary conditions?
    'bitres' : 'float32'                 ,                        # matrix element data type ('float32' vs 'float64')
    'in2m' : .0254                       ,                        # frequently used conversion
    'gpu_accel': True                    ,                        # use gpu to accelerate relevant computations 
    'torch': False
}

size_oom = np.log10(pdict['size'])
if np.sign(size_oom[0]) == -1:
    size_exp = np.floor(size_oom)
else:
    size_exp = np.ciel(size_oom)

pdict['res'] = (10**size_exp)*pdict['grid_res']
pdict['roi_res'] = pdict['res']*pdict['roi_mult']

if pdict['gpu_accel'] and pdict['torch']: 
    pdict['bitres'] = 'torch.' + pdict['bitres'] 
    pdict['device'] = 'cuda' 
else: 
    pdict['device'] = 'cpu'

pdict['res_exp'] = np.abs(np.log10(pdict['res'])).astype('int')
pdict['roi_res_exp'] = np.abs(np.log10(pdict['res'])+ np.log10(pdict['roi_mult'])).astype('int')
pdict['aspect'] = pdict['size'][0] == pdict['size'][1]
pdict['N'] = (np.round(pdict['size']*(1/pdict['res']) + 1)).astype('int')   # number of points sampled 1 dimension of simulation box
pdict['vec_shape'] = (pdict['N'].prod(),1)

# Sample parameters
pdict['optic'] =  {
        "diam" : 1.0*pdict['in2m'],
        "thickness" : .25*pdict['in2m'], 
        "z_com" : pdict['size'][1]/2,
        "sub_eps" : 3.82,                                        # dielectric constant for substrate (fused silica)
        "coat_eps" : 13.436,                                     # dielectric constant for coating material (AlGaAs / GaAs)
        "coat_thickness" : 9.5e-6     
}

pdict['optic']['sub_thickness'] = pdict['optic']['thickness'] - pdict['optic']['coat_thickness']

# CHOOSING ASSEMBLY CONFIGURATION
    # Parameters established to characterize the assembly configurations:
        # Front and back plate dimensions (usually disk diameters)
        # Central aperture diameter
        # Plate positioning along the beam axis with respect to simulation size center
        # Maximum AC voltage sent on respective plates

maxhva_settings = {
        "SVR350" : 210,                                         # [Vpk]
        "TREK2220" : 220,                                       # [Vpk]
        "TREK5/80" : 1000,                                      # [Vpk]
        "TREK10/10B-HS" : 1040                                  # [Vpk]
} 

if pdict['assembly'] == 0 or pdict['assembly'] == 1:
    # Setting front and back plate params (including spacing between them and voltage on respective plates)
    # This assembly has an assortment of 3d printed spacer components
    pdict['HVA'] = "SVR350"
    pdict['mount_zdims'] = {
        "back_ring" : 1e-3,                                    # +/- 2e-4 [m]
        "sample_holder" : 9e-3,                                # +/- 2e-4 [m]
        "electrode_brace": 3e-3 ,                              # +/- 2e-4 [m]
        "electrode_backing": 2e-3                              # +/- 2e-4 [m]
    }
    pdict['front_plate'] = {
        "diam" : 3* pdict['in2m'],                             # diameter of plate [m]
        "hole_diam" : 3e-3,                                    # aperture diameter [m]
        "thickness" : 1.5e-3,
        "zpos" : pdict['size'][1]/2 + pdict['mount_zdims']["sample_holder"]/2,         # location of plate surface (com) [m]
        "voltage" : maxhva_settings[pdict['HVA']]                       # Voltage on front plate [V]
    }
    pdict['back_plate'] = {
        "diam" : 3*pdict['in2m'], 
        "hole_diam" : 3e-3,
        "thickness" : 1.5e-3, 
        "zpos" : pdict['size'][1]/2 - pdict['mount_zdims']["sample_holder"]/2,
        "voltage" : - maxhva_settings[pdict['HVA']]
    } 
elif pdict['assembly'] == 2 : 
    # Overall thickness was approximately .5 inches with a .125 inch lip on one end and .125 gap on the other end of sample.
    # Once the sample was dropped into the mount with the surface hugging the PVC lip, it was held down with a nylon set screw (with a rubberized tip)
    # This plate used is an aluminum rectangular plate (will incorporate cartesian coordinates into program soon)
    pdict['HVA'] = "SVR350"
    pdict['mount_zdims'] = {
        "sample_holder" : .5*pdict['in2m']
    }
    if pdict['coords'] == 'cartesian':
        pdict['front_plate'] = {
            "diam" : 0.02794,                                 # diameter of plate [m]
            "hole_diam" : 3e-3,                               # aperture diameter [m]
            "thickness" : 1.27e-3, 
            "zpos" : pdict['size'][1]/2 + pdict['mount_zdims']["sample_holder"]/2,    # location of plate surface (com) [m]
            "voltage" : maxhva_settings[pdict['HVA']]/2                               # Voltage on front plate [V] (MAX value for associated HVA)
        }
        pdict['back_plate'] = {
            "diam" : 0.02794,
            "hole_diam" : 3e-3,
            "thickness" : 1.27e-3, 
            "zpos" : pdict['size'][1]/2 - pdict['mount_zdims']["sample_holder"]/2,
            "voltage" : - maxhva_settings[pdict['HVA']]/2
        }
    elif pdict['coords'] == 'cylindrical':
        pdict['front_plate'] = {
            "diam" : 0.02794,                                 # diameter of plate [m]
            "hole_diam" : 3e-3,                               # aperture diameter [m]
            "thickness" : 1.27e-3, 
            "zpos" : pdict['size'][1]/2 + pdict['mount_zdims']["sample_holder"]/2,    # location of plate surface (com) [m]
            "voltage" : maxhva_settings[pdict['HVA']]/2                                  # Voltage on front plate [V]
        }
        pdict['back_plate'] = {
            "diam" : 0.02794,
            "hole_diam" : 3e-3,
            "thickness" : 1.27e-3, 
            "zpos" : pdict['size'][1]/2 - pdict['mount_zdims']["sample_holder"]/2,
            "voltage" : maxhva_settings[pdict['HVA']]/2  
        }
        
elif pdict['assembly'] == 3 :
    #Set front and back plate params
    pdict['HVA'] =  "TREK10/10B-HS"
    pdict['mount_zdims'] = {
        "total_zthickness" : 25.94e-3 ,                         # holds both sample and both electrodes [m]
        "sample_holder" : 6.94e-3                              # width of lip that separates sample from electrodes [m] 
    }
    pdict['front_plate'] = {
        "diam" : 31.5e-3,                                       # diameter of plate [m]
        "hole_diam" : 3e-3,                                  # aperture diameter [m]
        "thickness" : 9.66e-3, 
        "zpos" : pdict['size'][1]/2 + (pdict['mount_zdims']["sample_holder"]/2),      # location of plate surface (com) [m]
        "voltage" : maxhva_settings[pdict['HVA']]/2                                     # Voltage on front plate [V]
    }
    pdict['back_plate'] = {
        "diam" : 31.5e-3,
        "hole_diam" : 3e-3,
        "thickness" : 9.66e-3, 
        "zpos" : pdict['size'][1]/2 - (pdict['mount_zdims']["sample_holder"]/2),
        "voltage" : -maxhva_settings[pdict['HVA']]/2
    }
    
elif pdict['assembly'] == 4 :
    #Set front and back plate params
    pdict['mount_zdims'] = {
        "total_zthickness" : 25.94e-3 ,                         # holds both sample and both electrodes [m]
        "sample_holder" : 6.94e-3                              # width of lip that separates sample from electrodes [m] 
    }
    pdict['front_plate'] = {
        "diam" : 31.5e-3,                                       # diameter of plate [m]
        "hole_diam" : 3e-3,                                  # aperture diameter [m]
        "thickness" : 9.66e-3, 
        "zpos" : pdict['size'][1]/2 + (pdict['mount_zdims']["sample_holder"]/2),      # location of plate surface (com) [m]
        "voltage" :  maxhva_settings[pdict['HVA']]/2                                      # Voltage on front plate [V]
    }
    pdict['back_plate'] = {
        "diam" : 31.5e-3,
        "hole_diam" : 3e-3,
        "thickness" : 9.66e-3, 
        "zpos" : pdict['size'][1]/2 - (pdict['mount_zdims']["sample_holder"]/2),
        "voltage" : - maxhva_settings[pdict['HVA']]/2 
    }
    
pdict['cap_params'] = {
        "area" : np.pi*((pdict['front_plate']['diam']/2.0)**2), 
        "d_air": (pdict['mount_zdims']['sample_holder']-pdict['optic']['thickness'])/2.0, 
        "air_eps" : 1.0006
    }

pdict['cap_params']['cap_div_area'] = (pdict['optic']['sub_eps']*pdict['optic']['coat_eps']*pdict['cap_params']['air_eps'])/((2.0*pdict['optic']['sub_eps']*pdict['optic']['coat_eps']*pdict['cap_params']['d_air']) + (pdict['optic']['sub_eps']*pdict['cap_params']['air_eps']*pdict['optic']['coat_thickness']) + (pdict['optic']['coat_eps']*pdict['cap_params']['air_eps']*(pdict ['optic']['sub_thickness'])))
pdict['cap_params']['capacitance'] = pdict['cap_params']['cap_div_area']*pdict['cap_params']['area']

# system location params / metadata
pdict['loc_params'] = {
    'center of optic' : {
        'cross_section_coord' : 'z',
        'rho' : 0,
        'z1_bound' : pdict['front_plate']['zpos'],
        'z2_bound' : pdict['back_plate']['zpos']
    },
    'edge of hole' : {
        'cross_section_coord' : 'z',
        'rho' : pdict['front_plate']['hole_diam'],
        'z1_bound' : pdict['front_plate']['zpos'],
        'z2_bound' : pdict['back_plate']['zpos']
    },
    'edge of optic' : {
        'cross_section_coord' : 'z',
        'rho' : pdict['optic']['diam'] /2,
        'z1_bound' : pdict['front_plate']['zpos'],
        'z2_bound' : pdict['back_plate']['zpos']
    },
    'edge of plate' : {
        'cross_section_coord' : 'z',
        'rho' : pdict['front_plate']['diam']/2,
        'z1_bound' : pdict['front_plate']['zpos'],
        'z2_bound' : pdict['back_plate']['zpos']
    },
    'halfway out on optic' : {
        'cross_section_coord' : 'z',
        'rho' : pdict['optic']['diam']/4,
        'z1_bound' : pdict['front_plate']['zpos'],
        'z2_bound' : pdict['back_plate']['zpos']
    },
    'front of plate' : {
        'cross_section_coord' : 'rho',
        'rho1_bound' : 0,
        'rho2_bound' : pdict['size'][0],
        'z' : pdict['back_plate']['zpos']
    },
    'front of optic' : {
        'cross_section_coord' : 'rho',
        'rho1_bound' : 0,
        'rho2_bound' : pdict['size'][0],
        'z' : pdict['optic']['z_com'] + pdict['optic']['thickness']/2
    },
    'middle of optic' : {
        'cross_section_coord' : 'rho',
        'rho1_bound' : 0,
        'rho2_bound' : pdict['size'][0],
        'z' : pdict['optic']['z_com']
    },
    'back of optic' : {
        'cross_section_coord' : 'rho',
        'rho1_bound' : 0,
        'rho2_bound' : pdict['size'][0],
        'z' : pdict['optic']['z_com'] - pdict['optic']['thickness']/2
    },
    'back plate' : {
        'cross_section_coord' : 'rho',
        'rho1_bound' : 0,
        'rho2_bound' : pdict['size'][0],
        'z' : pdict['back_plate']['zpos']
    }
}

pdict['roi_center'] = np.array([0, pdict['loc_params']['front of plate']['z']])