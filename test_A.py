import xarray as xr
import pdb
import numpy as np
A_path = r'/home/master/shujw/CSRMASCON/outdata/world_co_matv1.nc'
mask_path = r'/home/master/QK/pointmass/results/Greenland_mask/Greenland_mask.nc'
disgra_path = r'/home/master/shujw/CSRMASCON/outdata/dg/worldwideDisGravity.nc'
A_dataset = xr.open_dataset(A_path)
A=A_dataset['co_mat'].values
mask_dataset = xr.open_dataset(mask_path)
mask = mask_dataset['mask'].values
mask = np.squeeze(mask)
index = np.where(mask==1)
index = np.squeeze(index)
A_new = A[index,:]
A_new1 = A_new[:,index]
disgra_dataset = xr.open_dataset(disgra_path)
print(disgra_dataset.data_vars)
disgravity = disgra_dataset['disgravity'].values
disgravity_region = disgravity[:,index]
num_obs,num_unknown = np.shape(A_new1)
NR = np.eye(num_unknown)*1e-18
x = np.linalg.inv(A_new1.T@A_new1+NR)@A_new1.T@disgravity_region[10,:]

pdb.set_trace()