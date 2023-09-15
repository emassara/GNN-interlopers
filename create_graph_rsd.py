import numpy as np
import os
import h5py
import torch
from torch_geometric.data import Data, DataLoader
import scipy.spatial as SS

'''
Create graphs from sub-boxes 
'''

######################## INPUT ######################

# the displacement 
delta_d = 97.

#########################################

def create_edges(pos, r):
    kd_tree = SS.KDTree(pos)
    edge_index = kd_tree.query_pairs(r, output_type="ndarray")
    # Add reverse pairs
    reversepairs = np.zeros((edge_index.shape[0], 2))
    for i, edge in enumerate(edge_index):
        reversepairs[i] = np.array([edge[1], edge[0]])
    edge_index = np.append(edge_index, reversepairs, 0)
    edge_index = edge_index.astype(int)
    # Use Pytorch Geometric format
    edge_index = edge_index.reshape((2,-1))
    return edge_index


# Create dataset
"""
r ------------------> link length
mode ---------------> 'train', 'valid', 'test' or 'all'
sims ---------------> number of realizations to consider
seed ---------------> seed to shuffle the realizations
batch_size ---------> batch size
shuffle ------------> whether to shuffle the data or not
mode_edge ----------> 'all', 'rperp', 'rpar', 'rpar_rperp'
"""
def create_dataset(r, mode, sims, seed, batch_size, shuffle, snapnum = 2, cosmo = 'fiducial',
                   mode_edge='all'):
    # Create data container
    dataset = []
    
    z_min = 0
    z_max = 1000.
    
    delta_xy = 150.
    n_x = 6
    n_y = 6
    
    if   mode=='train':  size, offset = int(sims*0.8), int(sims*0.0)
    elif mode=='valid':  size, offset = int(sims*0.1), int(sims*0.8)
    elif mode=='test':   size, offset = int(sims*0.1), int(sims*0.9)
    elif mode=='all':    size, offset = int(sims*1.0), int(sims*0.0)
    else:                raise Exception('Wrong name!')

    print(mode_edge)
    
    indexes = np.arange(sims) #only shuffle realizations
    np.random.seed(seed)
    np.random.shuffle(indexes)
    indexes = indexes[offset:offset+size]
        
    # Get catalog file
    for realiz in indexes: 
        for x_id in range(n_x):
            x_min = x_id*delta_xy
            x_max = x_min+delta_xy
            for y_id in range(n_y):
                y_min = y_id*delta_xy
                y_max = y_min+delta_xy
                
                for entry in os.scandir('measurements/%s'%(cosmo)):
                    if entry.name.startswith('realiz-%i-%s_x-%s-%s_y-%s-%s_z-%s-%s_fi'%(realiz,snapnum,x_min,x_max,y_min,
                                                                                        y_max,z_min,z_max)):
                        x = entry.name.split("-")
                        f_i = x[9].split("_")[0]
        
                        filename = 'measurements/%s/realiz-%i-%s_x-%s-%s_y-%s-%s_z-%s-%s_fi-%s_dd-%s.hdf5'%(cosmo,realiz,snapnum,x_min,x_max,y_min,y_max,z_min,z_max,f_i,delta_d)
                        
                        # Create graph and append to data container
                        dataset.append(create_graph(filename, snapnum, cosmo, r, x_min, x_max, y_min, y_max, f_i, mode_edge))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


def create_graph(filename, snapnum, cosmo, r, x_min, x_max, y_min, y_max, f_i, mode_edge):
    # Load catalog 
    f = h5py.File(filename, 'r')
    pos  = np.array(f['pos']) #Mpc/h
    vel  = np.array(f['vel']) #km/s

    # get rsd
    if cosmo == 'fiducial':
        Omega_m = 0.3175
        Hubble = 100*(1-Omega_m+Omega_m*(1+redshift)**3)**0.5
    else:
        print('wrong cosmology')
        sys.exit()
    redshift = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]
    BoxSize = 1000.
    factor    = (1.0 + redshift)/Hubble
    pos[:,2] = (pos[:,2] + vel[:,2]*factor)%BoxSize
    
    # create edges
    edge_index = create_edges(pos, r)
    col,row = edge_index
    # create edge attributes
    p_p = [(x_min+x_max)/2.0, (y_min+y_max)/2.0] 
    r_par = torch.tensor(abs(pos[col,2]-pos[row,2])/r,dtype=torch.float32).unsqueeze(dim=1)
    r_perp =torch.tensor((((pos[col,0]-pos[row,0])**2+(pos[col,1]-pos[row,1])**2)**0.5)/r,dtype=torch.float32).unsqueeze(dim=1)
    p_i = (pos[col,0:2]-p_p)/(np.sum((pos[col,0:2]-p_p)**2,axis=1)[None,:].T)**0.5
    p_j = (pos[row,0:2]-p_p)/(np.sum((pos[row,0:2]-p_p)**2,axis=1)[None,:].T)**0.5
    theta = torch.tensor(np.sum(p_i*p_j,axis=1),dtype=torch.float32).unsqueeze(dim=1)

    if mode_edge == 'all':
        edge_attr = torch.cat([r_par,r_perp,theta],dim=1)
    elif mode_edge == 'rpar':
        edge_attr = r_par
    elif mode_edge == 'rperp':
        edge_attr = r_perp
    elif mode_edge == 'theta':
        edge_attr = theta
    elif mode_edge == 'rpar_rperp':
        edge_attr = torch.cat([r_par,r_perp],dim=1)

        
    x = torch.tensor(np.zeros(len(pos)),dtype=torch.float32).unsqueeze(dim=1)
    y = torch.tensor([float(f_i)], dtype=torch.float32)
    y = y.unsqueeze(dim=1)
    # Construct graph
    graph = Data(x          = x,
                 y          = y, #torch.tensor([float(f_i)], dtype=torch.float32),
                 edge_index = torch.tensor(edge_index, dtype=torch.long),
                 edge_attr  = edge_attr)
                 
    return graph

