import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ppt2latexsubfig')
plt.rcParams["font.family"] = "Times New Roman"

# Open the HDF5 file in read mode
file_path1 = 'Aplus_mod_algaas.hdf5'
file_path2 = 'Aplus_mod_silicatitantala.hdf5'

def hdf5_check(file_path):
    with h5py.File(file_path, 'r') as file:
        # Function to recursively print the HDF5 dataset hierarchy
        def print_hdf5_item(name, obj):
            # name is in path format like /group1/group2/dataset
            if isinstance(obj, h5py.Group):
                # Do something like creating a dictionary entry
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                # Access the dataset values
                data = obj[()]
                print(f"Dataset: {name} - Shape: {data.shape}")
    
        # Print all root level object names (aka keys)
        print("Keys:", list(file.keys()))
    
        # Recursively print the hierarchy
        file.visititems(print_hdf5_item)

hdf5_check(file_path1)

aplus_mod_algaas = h5py.File(file_path1,'r')
aplus_mod_silicatitantala = h5py.File(file_path2, 'r')

list(aplus_mod_algaas['A+_mod'])

freq = aplus_mod_algaas['Freq'][:]
CB_algaas = aplus_mod_algaas['A+_mod']['budget']['CoatingBrownian']['PSD'][:]
budgetAL = aplus_mod_algaas['A+_mod']['PSD'][:]
CB_silicatitantala = aplus_mod_silicatitantala['A+_mod']['budget']['CoatingBrownian']['PSD'][:]
budgetST = aplus_mod_silicatitantala['A+_mod']['PSD'][:]

sqrt_CBAL = CB_algaas**(1.0/2.0)
sqrt_CBST = CB_silicatitantala**(1.0/2.0)

sqrt_budgAL = budgetAL**(1.0/2.0)
sqrt_budgST = budgetST**(1.0/2.0)

aplus_mod_algaas['A+_mod']['budget']


lw=8
plt.loglog(freq,sqrt_budgAL, label ='A sharp (Total noise)', color='C0', linewidth= lw)
plt.loglog(freq,sqrt_budgST, label = 'A plus (Total noise)', color ='C1', linewidth= lw)
plt.loglog(freq, sqrt_CBAL, label = 'A sharp (Coating Brownian noise)', alpha=.5, linestyle=':', color='C0', linewidth=lw)
plt.loglog(freq, sqrt_CBST, label = 'A plus (Coating Brownian noise)', alpha=.5, linestyle=':', color='C1', linewidth=lw)
plt.legend(prop={'size': 50})
plt.ylabel('[1/sqrt{Hz}]')
plt.xlabel('frequency [Hz]')

plt.savefig('aplus_CTN_compare.pdf',dpi=300, format='pdf', bbox_inches='tight')


