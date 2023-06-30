import numpy as np
import sys,os

import readgadget
import readfof
import h5py


'''
Assume the LoS to be along the z axis 
'''

# mass cut but fixed Nhalos
def contaminated_catM_Nfixed(pos_sel_xy,vel_sel_xy,mass_sel_xy,z_min,z_max,f_i,delta_d,realiz,snapnum,cosmo,N_halos,N_part):

    # introduce interlopers (fraction of the actual number of halos - sometimes ~ 15% they are less than N_halos)
    id_part = np.arange(len(pos_sel_xy))
    interloper = np.zeros(len(pos_sel_xy))
    N_int = int(len(pos_sel_xy)*f_i)
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
    mass = mass_sel_xy[sel_z]
    interl = interloper[sel_z]

    # write to h5py file
    fout = 'measurements/%s/Mh_cut_Nfixed%s/realiz-%i-%s_x-%s-%s_y-%s-%s_z-%s-%s_fi-%s_Nh-%d_Npmin-%d_dd-%s.hdf5'%(cosmo,label_new,realiz,snapnum,x_min,x_max,y_min,y_max,z_min,z_max,f_int[i_box],N_halos,N_part,delta_d)
    f = h5py.File(fout, 'w')
    f.create_dataset('pos', data=np.asarray(pos))
    f.create_dataset('vel', data=np.asarray(vel))
    f.create_dataset('int', data=np.asarray(interl))
    f.create_dataset('mass', data=np.asarray(mass))
    f.close()
    print(fout)

# catalogs specifying the mass as well
def contaminated_catM(pos_sel_xy,vel_sel_xy,mass_sel_xy,z_min,z_max,f_i,delta_d,realiz,snapnum,cosmo,N_halos): #M_min):
    
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
    mass = mass_sel_xy[sel_z]
    interl = interloper[sel_z]

    # write to h5py file
    #fout = 'measurements/%s/Mh_cut/realiz-%i-%s_x-%s-%s_y-%s-%s_z-%s-%s_fi-%s_Mhmin-%.4E_dd-%s.hdf5'%(cosmo,realiz,snapnum,x_min,x_max,y_min,y_max,z_min,z_max,f_int[i_box],M_min,delta_d)
    fout = 'measurements/%s/Mh_cut/realiz-%i-%s_x-%s-%s_y-%s-%s_z-%s-%s_fi-%s_Nh-%d_dd-%s.hdf5'%(cosmo,realiz,snapnum,x_min,x_max,y_min,y_max,z_min,z_max,f_int[i_box],N_halos,delta_d)
    f = h5py.File(fout, 'w')
    f.create_dataset('pos', data=np.asarray(pos))
    f.create_dataset('vel', data=np.asarray(vel))
    f.create_dataset('int', data=np.asarray(interl))
    f.create_dataset('mass', data=np.asarray(mass))
    f.close()


# catalogs without mass specified (used to train with same halo bias/mass
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

cosmo = 'LH' #'fiducial' 'LH'
BoxSize = 1000.0

snapnum = 2
delta_d = 97.
mass_cut = True

z_min = 0
z_max = 1000.
delta_x = 250. #1000. #250.
delta_y = delta_x

N_halos = 4500 #2500

label_new = '_new' # get additional realizations for LHC, N_halos = 4500 and delta_x = 250
    
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
    
if cosmo =='fiducial':
    n_realiz = 100
    root_halos = '/home/emassara/projects/rrg-wperciva/Quijote_halos/%s'%(cosmo)
elif cosmo == 'LH':
    n_realiz = 2000
    root_halos = '/home/emassara/projects/rrg-wperciva/Quijote_Halos_LH/'
    f_in = 'latin_hypercube_params.txt'
    Omega_m,Omega_b,h,ns,s8 = np.loadtxt(f_in,unpack=True)
    sel = np.where((Omega_m>0.18)&(Omega_m<0.42)&(Omega_b>0.038)&(Omega_b<0.062)&(h>0.58)&(h<0.82)&(ns>0.88)&(ns<1.12)&(s8>0.68)&(s8<0.92))[0]
    print(len(sel))


i_small = 0
i_30 = 0
i_30_small = 0

for realiz in sel: #range(0,n_realiz):

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
    #N_halos = np.random.randint(600,4200,72) #190000,23000,72)
    #Mh_min = np.random.uniform(13.1,14.,72)
    #Mh_min = 10**Mh_min
    #number of DM particles in 1 halo
    N_part = np.random.randint(20,31,72)  #np.random.randint(19,30,72) #np.random.randint(20,31,72) for fixed cosmo
    i_box = 0
    
    for x_id in range(n_x):
        x_min = x_id*delta_x
        x_max = x_min+delta_x
        for y_id in range(n_y):
            y_min = y_id*delta_y
            y_max = y_min+delta_y
            #if x_id ==0 and y_id == 0:
            #    i_box+=1
            #    continue
            if mass_cut:
                """# for Mhcut
                sel_halos = np.where((pos_h[:,0]>x_min)&(pos_h[:,0]<x_max)&(pos_h[:,1]>y_min)&(pos_h[:,1]<y_max)&(mass>Mh_min[i_box]))[0]
                pos_sel_xy = pos_h[sel_halos]
                vel_sel_xy = vel_h[sel_halos]
                mass_sel_xy = mass[sel_halos]
                contaminated_catM(pos_sel_xy,vel_sel_xy,mass_sel_xy,z_min,z_max,f_int[i_box],delta_d,realiz,snapnum,Mh_min[i_box])
                """
                
                """# for Nhalo cut (most N massive halos)
                sel_halos = np.where((pos_h[:,0]>x_min)&(pos_h[:,0]<x_max)&(pos_h[:,1]>y_min)&(pos_h[:,1]<y_max))[0]
                pos_sel_xy = pos_h[sel_halos]
                vel_sel_xy = vel_h[sel_halos]
                mass_sel_xy = mass[sel_halos]
                if N_halos[i_box] > len(mass_sel_xy):
                    N_halos[i_box]=len(mass_sel_xy)
                sel_halos_2 = np.argsort(mass_sel_xy)
                sel_halos_2 = sel_halos_2[::-1][:N_halos[i_box]]
                pos_sel_xy = pos_sel_xy[sel_halos_2]
                vel_sel_xy = vel_sel_xy[sel_halos_2]
                mass_sel_xy = mass_sel_xy[sel_halos_2]
                contaminated_catM(pos_sel_xy,vel_sel_xy,mass_sel_xy,z_min,z_max,f_int[i_box],delta_d,realiz,snapnum,cosmo,
                N_halos[i_box])
                """

                # for Mhcut at fixed Nhalo
                sel_halos = np.where((pos_h[:,0]>x_min)&(pos_h[:,0]<x_max)&(pos_h[:,1]>y_min)&(pos_h[:,1]<y_max)&(Npart>=N_part[i_box]))[0]
                pos_sel_xy = pos_h[sel_halos]
                vel_sel_xy = vel_h[sel_halos]
                mass_sel_xy = mass[sel_halos]
                id_part = np.arange(len(mass_sel_xy))
                np.random.shuffle(id_part)
                sel_halos_2 = id_part[:N_halos]
                pos_sel_xy = pos_sel_xy[sel_halos_2]
                vel_sel_xy = vel_sel_xy[sel_halos_2]
                mass_sel_xy = mass_sel_xy[sel_halos_2]

                """
                if N_part[i_box] ==30:
                    i_30 +=1
                if N_part[i_box] ==30:
                    if len(mass_sel_xy) != N_halos:
                        i_30_small +=1
                if len(mass_sel_xy) != N_halos:
                    print(realiz, N_part[i_box], len(mass_sel_xy))
                    i_small += 1
                """
                
                contaminated_catM_Nfixed(pos_sel_xy,vel_sel_xy,mass_sel_xy,z_min,z_max,f_int[i_box],delta_d,realiz,snapnum,cosmo,
                                         N_halos,N_part[i_box])
            else:
                sel_halos = np.where((pos_h[:,0]>x_min)&(pos_h[:,0]<x_max)&(pos_h[:,1]>y_min)&(pos_h[:,1]<y_max))[0]
                pos_sel_xy = pos_h[sel_halos]
                vel_sel_xy = vel_h[sel_halos]
                contaminated_cat(pos_sel_xy,vel_sel_xy,z_min,z_max,f_int[i_box],delta_d,realiz,snapnum,cosmo)
            
            i_box+=1
            #contaminated_cat(pos_sel_xy,vel_sel_xy,500.,1000.,f_int[i_box],realiz,snapnum)
            #i_box+=1
print(i_box)
sys.exit()            

print(i_small)
print(i_30_small)
print(i_30)
