import numpy as np
import os
import h5py
import torch
from torch_geometric.data import Data, DataLoader
import scipy.spatial as SS

print('using create_graph_rsd.py')

######################## INPUT ######################

#cosmo = 'fiducial'
#BoxSize = 1000.0

#snapnum = 2
delta_d = 97.

#z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

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
        Hubble = 67.11
    else:
        print('wrong cosmology')
        sys.exit()
    redshift = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]
    BoxSize = 1000.
    factor    = (1.0 + redshift)/Hubble
    pos[:,2] = (pos[:,2] + vel[:,2]*factor)%BoxSize
    
    #i_h  = np.array(f['int']) # 1: interloper
    # create edges
    edge_index = create_edges(pos, r)
    col,row = edge_index
    # create edge attributes
    p_p = [(x_min+x_max)/2.0, (y_min+y_max)/2.0] # is this necessary for the angles? 
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

"""
# Create dataloaders for train, valid, test
def create_loaders(dataset, seed, batch_size):

    np.random.seed(seed)
    np.random.shuffle(dataset)

    size_train, offset_train = int(0.8 * len(dataset)), 0
    size_valid, offset_valid = int(0.1 * len(dataset)), int(0.8 * len(dataset))
    size_test,  offset_test  = int(0.1 * len(dataset)), int(0.9 * len(dataset))

    train_dataset = dataset[offset_train:size_train]
    valid_dataset = dataset[offset_valid:offset_valid+size_valid]
    test_dataset = dataset[offset_test:offset_test+size_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

"""

#graph = create_dataset(2, 2, 10)
"""print(graph)
print(graph[0].y)
print(graph[0].edge_index)
print(graph[0].edge_attr)
"""
#create_dataset(10, 'test', 100, 4, 4, True, 2400, snapnum = 2, cosmo = 'fiducial', large_r = 'False')
"""
graph = create_dataset(25, 'test', 10, 4, 1, False, 2400,150., mode_edge='all')
old = []
for data in graph:
    old.append(len(data.edge_index.T))
graph = create_dataset(25, 'test', 10, 4, 1, False, 4500,250., mode_edge='all')
new = []
for data in graph:
    new.append(len(data.edge_index.T))
old = np.array(old)
new = np.array(new)
print(new/old[:32])
"""
