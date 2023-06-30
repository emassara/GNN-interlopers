import numpy as np
import sys,os

import readgadget
import readfof
import h5py


'''
Crop sub-boxes of wanted size along x and y direction
Introduce interlopers
Assume the LoS to be along the z axis 
'''

# catalogs used to train with same halo bias/mass
def contaminated_cat(pos_sel_xy,vel_sel_xy,z_min,z_max,f_i,delta_d,realiz,snapnum,cosmo):
    
    N_halos = len(pos_sel_xy)
    id_part = np.arange(N_halos)
    interloper = np.zeros(N_halos)
    N_int = int(N_halos*f_i)
    np.random.shuffle(id_part)
    sel_int = id_part[:N_int]
    pos_z = pos_sel_xy[:,2]
    pos_z[sel_int] = (pos_z[sel_int] + delta_d)%BoxSize
    interloper[sel_int] = 1 # flag interloper
    sel_z = np.where((pos_z>z_min)&(pos_z<z_max))[0]
    pos = np.zeros((len(sel_z),3))
    pos[:,0] = pos_sel_xy[sel_z,0]
    pos[:,1] = pos_sel_xy[sel_z,1]
    pos[:,2] = pos_z[sel_z]
    vel = vel_sel_xy[sel_z]
    interl = interloper[sel_z]
    # write to h5py file
    fout = 'measurements/%s/realiz-%i-%s_x-%s-%s_y-%s-%s_z-%s-%s_fi-%s_dd-%s.hdf5'%(cosmo,realiz,snapnum,x_min,x_max,y_min,y_max,z_min,z_max,f_int[i_box],delta_d)
    f = h5py.File(fout, 'w')
    f.create_dataset('pos', data=np.asarray(pos))
    f.create_dataset('vel', data=np.asarray(vel))
    f.create_dataset('int', data=np.asarray(interl))
    f.close()
    
######################## INPUT ######################       

cosmo = 'fiducial'
BoxSize = 1000.0

snapnum = 2
delta_d = 97. # interloper displacement

n_realiz = 100
root_halos = '/home/emassara/projects/rrg-wperciva/Quijote_halos/%s'%(cosmo)

z_min = 0
z_max = 1000.
# size along x and y direction
delta_x = 150. 
delta_y = delta_x
    
#########################################
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

if delta_x == 150.:
    n_x = 6
    n_y = 6
elif delta_x == 200.:
    n_x = 5
    n_y = 5
elif delta_x == 250.:
    n_x = 4
    n_y = 4
elif delta_x == 1000.:
    n_x = 1
    n_y = 1
    

for realiz in range(0,n_realiz):

    # read halo catalog
    halodir = '%s/%d'%(root_halos,realiz)
    FoF = readfof.FoF_catalog(halodir, snapnum, long_ids=False,
                              swap=False, SFR=False, read_IDs=False)
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h                         
    vel_h = FoF.GroupVel*(1.0+z)        #Halo peculiar velocities in km/s
    Npart = FoF.GroupLen                #Number of CDM particles in the halo

    ######## Selection ########
    f_int = np.random.uniform(0,0.2,72)
    
    for x_id in range(n_x):
        x_min = x_id*delta_x
        x_max = x_min+delta_x
        for y_id in range(n_y):
            y_min = y_id*delta_y
            y_max = y_min+delta_y

            sel_halos = np.where((pos_h[:,0]>x_min)&(pos_h[:,0]<x_max)&(pos_h[:,1]>y_min)&(pos_h[:,1]<y_max))[0]
            pos_sel_xy = pos_h[sel_halos]
            vel_sel_xy = vel_h[sel_halos]
            contaminated_cat(pos_sel_xy,vel_sel_xy,z_min,z_max,f_int[i_box],delta_d,realiz,snapnum,cosmo)
