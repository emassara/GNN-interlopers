import numpy as np
import GNN_arch, create_graph_rsd
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import scipy.spatial as SS
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm, GCNConv
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


################################# Objective Function #############################
class objective(object):
    def __init__(self, sims, r_min, r_max, min_layers, max_layers, device, n_max,
                 n_min, num_epochs, dim_out, seed, batch_size, snapnum, mode_edge):

        self.sims               = sims
        self.min_layers         = min_layers
        self.max_layers         = max_layers
        self.r_min              = r_min
        self.r_max              = r_max
        self.device             = device
        self.num_epochs         = num_epochs
        self.n_max              = n_max
        self.n_min              = n_min
        self.dim_out            = dim_out
        self.seed               = seed
        self.batch_size         = batch_size
        self.mode_edge          = mode_edge
        self.snapnum            = snapnum
    
    def __call__(self, trial):
        
        # Files for saving results and best model
        f_text_file   = 'models/snap%i/losses_LFI/%s_%d.txt'%(self.snapnum,study_name,trial.number)
        f_best_model  = 'models/snap%i/models_LFI/%s_%d.pt'%(self.snapnum,study_name,trial.number)  
        

        # Generate the model
        n_layers = trial.suggest_int("n_layers", self.min_layers, self.max_layers)
        hid_channels = trial.suggest_int("hid_channels", self.n_min, self.n_max)
        if self.mode_edge == 'all':
            edge_in=3
        elif self.mode_edge == 'rpar_rperp':
            edge_in=2
        else:
            edge_in=1
        model = GNN_arch.GNN(trial, n_layers, self.dim_out, 1, hid_channels, edge_in).to(device)

        # Define optimizer
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)

        r = trial.suggest_float("r", self.r_min, self.r_max)#, log=True) 

        print('chosen values',n_layers,hid_channels,r)
        
        # Create Dataloaders
        torch.manual_seed(self.seed)
        train_loader = create_graph_rsd.create_dataset(r,'train', self.sims, self.seed,self.batch_size,
                                                       shuffle=True, snapnum=self.snapnum,mode_edge=self.mode_edge)
        valid_loader = create_graph_rsd.create_dataset(r,'valid', self.sims, self.seed,self.batch_size,
                                                       shuffle=True, snapnum=self.snapnum,mode_edge=self.mode_edge)
        
        
        # Train the model
        min_valid = 1e5
        for epoch in range(num_epochs):
            model.train()
            train_loss1, train_loss = torch.zeros(1).to(device), 0.0
            train_loss2, points     = torch.zeros(1).to(device), 0
            for data in train_loader:
                data = data.to(device=device)
                target = (data.y-0.1)/0.06
                target = target[:,0]
                bs    = target.shape[0]
                optimizer.zero_grad()
                output = model(data)
                loss1 = torch.mean((output[:,0] - target)**2)
                loss2 = torch.mean(((output[:,0] - target)**2 - output[:,1]**2)**2)
                loss  = torch.log(loss1) + torch.log(loss2)
                train_loss1 += loss1*bs
                train_loss2 += loss2*bs
                points      += bs
                loss.backward()
                optimizer.step()
            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
            train_loss = train_loss.item()
            
            # Validation of the model.
            model.eval() 
            valid_loss1, valid_loss = torch.zeros(1).to(device), 0.0
            valid_loss2, points     = torch.zeros(1).to(device), 0
            for data in valid_loader:
                data = data.to(device=device)
                target = (data.y-0.1)/0.06
                target = target[:,0]
                bs    = target.shape[0]
                output = model(data)
                loss1 = torch.mean((output[:,0] - target)**2)
                loss2 = torch.mean(((output[:,0] - target)**2 - output[:,1]**2)**2)
                loss  = torch.log(loss1) + torch.log(loss2)
                valid_loss1 += loss1*bs
                valid_loss2 += loss2*bs
                points      += bs

            valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
            valid_loss = valid_loss.item()

            if valid_loss<min_valid:
                min_valid = valid_loss
                torch.save(model.state_dict(), f_best_model)

            f = open(f_text_file, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
            f.close()

        return min_valid

##################################### INPUT #######################################
# Data Parameters
seed        = 4
snapnum     = 2
sims        = 100

# Training Parameters
num_epochs = 1000
batch_size = 4

# Architecture Parameters
n_min      = 1          # Minimum number of neurons in hidden layers
n_max      = 60         # Maximum number of neurons in hidden layers
min_layers = 1          # Minimum number of hidden layers
max_layers = 3          # Maximum number of hidden layers
dim_out    = 2          # Size of output = (f_i,sigma)
mode_edge  = 'all'      #'all', 'rperp', 'rpar', 'rpar_rperp'
r_min      = 5
r_max      = 30

# Optuna Parameters
n_trials   = 12
n_startup_trials = 50

study_name = 'GNN_RSD_batch%d_%s'%(batch_size,mode_edge)
n_jobs     = 1
storage    = 'sqlite:///%s.db'%(study_name)

############################## Start OPTUNA Study ###############################

# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
    
if __name__ == "__main__":

    objective = objective(sims, r_min, r_max, min_layers, max_layers, device, n_max,
                          n_min, num_epochs, dim_out, seed, batch_size, snapnum, mode_edge)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage,
                                load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs)

    print("  Number of finished trials: ", len(study.trials))
    
    trial = study.best_trial
    print("Best trial: number {}".format(trial.number))
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
