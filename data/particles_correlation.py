import numpy as np
from numpy import pi, sqrt, log, log10, cos, sin
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from lvmc.core.particle_lattice import ParticleLattice, Orientation
from lvmc.core.magnetic_field import MagneticField
from lvmc.core.flow import PoiseuilleFlow
from enum import Enum, auto
from typing import NamedTuple, List, Optional
import json
import h5py

from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from tqdm import tqdm
from utils import *
from rich import print
import sys

import time
from IPython.display import display, clear_output

# Parameters for ParticleLattice                                                                                                          
width = 50
height = 25
v0 = 100.0
v1 = 200
d = [0,1,2,3,4,5,6,7,8,9,10,12,15,18,21,25,30,35,40,45,49] 
#glist = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4]
glist = [1.8,2.0,2.2,2.4]
tmax = 500
density = 0.3

seed = 3179

def read_lattice(t0, tmax, g, v0, v1):
    flow_params = {"type": "poiseuille", "v1": v1}
    if flow_params["v1"] == 0:
        dt_flow = 2*tmax
        mydir = "./" 
        #mydir = "noflow_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')
        base_name = "noflow_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')
    else:
        dt_flow = 0.1/flow_params["v1"]
        mydir = "./" 
        #mydir = flow_params["type"]+"_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')+"_"+str(flow_params["v1"]).removesuffix('.0')
        base_name = flow_params["type"]+"_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')+"_"+str(flow_params["v1"]).removesuffix('.0')

    fname_stats = "stat_"+base_name+"seed%d"%(seed)+".txt"
    dt_stat = 0.1
    dt_dump_stat = 5
    dt_dump_field = 50

    # Initialize the Simulation (test if restart of not)                                                                                  
    if t0==0:
        print("Starting simulation from scratch with g = %g, v1 = %g" % (g,flow_params["v1"]))
        simulation = Simulation(g, v0, width=width, height=height, density=density, flow_params=flow_params, with_transport=False)
        obstacles = torch.zeros((height, width), dtype=torch.bool)
        obstacles[0, :] = True
        obstacles[-1, :] = True
        simulation.lattice.set_obstacles(obstacles)
    else:
        fname =  mydir+"fields_"+base_name+"_"+("%1.2f"%(t0-dt_dump_field))+"_"+("%1.2f"%t0)+"seed%d"%(seed) + ".h5"
        simulation = Simulation.init_from_file(fname)
    data_collector = DataCollector(simulation)
    
    return simulation.lattice


def find_particle_pairs_at_distance(lattice, d):
    pairs = []
    height, width = lattice.shape
    
    for y1 in range(1,height):
        for x1 in range(width):
            if lattice[y1, x1]:
                for y2 in range(y1,height):
                    for x2 in range(x1,width):
                        if lattice[y2, x2]:
                            if abs(y1 - y2) + abs(x1 - x2) == d:
                                pairs.append(((y1, x1), (y2, x2)))

    return pairs



def find_particle_pairs_at_distance_alongx(lattice, dx):
    pairs = []
    height, width = lattice.shape
    
    for y1 in range(1,height):
        for x1 in range(width):
            if lattice[y1, x1]:
                y2 = y1
                for x2 in range(x1, width):
                    if lattice[y2, x2]:
                        if abs(y1 - y2) + abs(x1 - x2) == dx:
                            pairs.append(((y1, x1), (y2, x2)))

    return pairs # returns all the couples at a distance 'd' along x (i.e. d = dx) in the lattice. 


def scalar_prod(y1,x1,y2,x2):
    sc = (int((lattice.orientation_map[y1,x1]==Orientation.UP))*(int(lattice.orientation_map[y2,x2]==Orientation.UP)) +
        int((lattice.orientation_map[y1,x1]==Orientation.UP))*(-int(lattice.orientation_map[y2,x2]==Orientation.DOWN)) + 
        int(-(lattice.orientation_map[y1,x1]==Orientation.DOWN))*(int(lattice.orientation_map[y2,x2]==Orientation.UP)) + 
        int(-(lattice.orientation_map[y1,x1]==Orientation.DOWN))*(-int(lattice.orientation_map[y2,x2]==Orientation.DOWN)) +
        int((lattice.orientation_map[y1,x1]==Orientation.RIGHT))*(int(lattice.orientation_map[y2,x2]==Orientation.RIGHT)) + 
        int((lattice.orientation_map[y1,x1]==Orientation.RIGHT))*(-int(lattice.orientation_map[y2,x2]==Orientation.LEFT)) +
        int(-(lattice.orientation_map[y1,x1]==Orientation.LEFT))*(int(lattice.orientation_map[y2,x2]==Orientation.RIGHT)) +
        int(-(lattice.orientation_map[y1,x1]==Orientation.LEFT))*(-int(lattice.orientation_map[y2,x2]==Orientation.LEFT)) )
    return sc



def Correlation(pairs):
    Corr = 0    
    for num_couple in range(len(pairs)): #could be only along 'x' or both 'x' and 'y'. 
        first_particle = 0
        second_particle = 1
        y1 = pairs[num_couple][first_particle][0]
        x1 = pairs[num_couple][first_particle][1]
        y2 = pairs[num_couple][second_particle][0]
        x2 = pairs[num_couple][second_particle][1]
        
        Corr += 1/len(pairs) * (scalar_prod(y1,x1,y2,x2))
        
    return Corr


def mean(lattice):
    mean_p = [0,0]
    for y1 in range(1,height):
        for x1 in range(width):
            if lattice.occupancy_map[y1, x1]:
                mean_p[1] += 1/Np * (int((lattice.orientation_map[y1,x1]==Orientation.UP)) - 
                                     int((lattice.orientation_map[y1,x1]==Orientation.DOWN)))
                mean_p[0] += 1/Np * (int((lattice.orientation_map[y1,x1]==Orientation.RIGHT)) - 
                                     int((lattice.orientation_map[y1,x1]==Orientation.LEFT)))

    return mean_p


flow_params = {"type": "poiseuille", "v1": v1}
lattice = read_lattice(t0 = 200, tmax = 500, g = 0.4, v0 = 100., v1 = 0)
# analyse the lattice 
Np = sum(sum(lattice.occupancy_map))
print('total number of particles in the lattice = %d'%(Np))
print('density = %.1f'%(Np/height/width))


times = np.arange(200,500,50)
#glist = [2.0,2.2,2.4] 


for g in glist:
    print('Looking at g = %g' %(g))
    C = np.zeros(len(d))
    count = np.zeros(len(d))
    Autocorr = np.zeros(len(d))
    mean_pt = [0,0]
    
    for t in times:
        print('Analyzing time %d' %(t))
        lattice = read_lattice(t0 = t, tmax = 500, g = g, v0 = 100., v1 = v1)

        #mean direction of particles in the lattice 
        mean_t = mean(lattice)
        mean_pt[0] += mean_t[0]
        mean_pt[1] += mean_t[1]
        
        for m in d:
            pairs = find_particle_pairs_at_distance_alongx(lattice.occupancy_map, m)
            print("Particle pairs at distance", m, 'along x direction ', ' = ', len(pairs))
            #print("Particle pairs at distance", m,' = ', len(pairs))
            Corr = Correlation(pairs)
            if len(pairs) and Corr != 0:
                count[d.index(m)] += 1
            C[d.index(m)] += Corr
            print(C)
        #plt.plot(d, C/count[d.index(m)], '.-')
    
    for m in d:
        if count[d.index(m)] != 0:
            C[d.index(m)] = C[d.index(m)] / count[d.index(m)] #average on time

    # time average of particles direction in the lattice 
    mean_pt[0] = mean_pt[0] / times.shape[0]
    mean_pt[1] = mean_pt[1] / times.shape[0]    
    #square of the mean
    mean_sq_pt = mean_pt[0]*mean_pt[0] + mean_pt[1]*mean_pt[1] 
    
    # calculate autocorrelation
    for m in d:
        Autocorr[d.index(m)] = (C[d.index(m)] - mean_sq_pt) / (1 - mean_sq_pt)
    
    if flow_params["v1"] == 0:
        dt_flow = 2*tmax
        base_name = "noflow_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')
    else:
        dt_flow = 0.1/flow_params["v1"]
        base_name = flow_params["type"]+"_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')+"_"+str(flow_params["v1"]).removesuffix('.0')

    array_to_save = ([d,C,Autocorr])
    array_to_save = np.array(array_to_save)
    array_to_save.shape
    np.savetxt("correlation/" + "Correlation_"+ base_name, array_to_save.T, delimiter = ' ', fmt = '%g')
