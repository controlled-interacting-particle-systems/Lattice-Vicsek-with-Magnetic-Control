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
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib

# Parameters for ParticleLattice                                                                                                           
width = 50
height = 25
v0 = 100.0
density = 0.3

seed = 3179

def read_lattice(t0, tmax, g, v0, v1):
    flow_params = {"type": "poiseuille", "v1": v1}
    if flow_params["v1"] == 0:
        dt_flow = 2*tmax
        mydir = "file_output/" 
        base_name = "noflow_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0') 
    else:
        dt_flow = 0.1/flow_params["v1"]
        mydir = "file_output/" 
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
        fname =  mydir+"fields_"+base_name+"_"+("%1.2f"%(t0-dt_dump_field))+"_"+("%1.2f"%t0)+"seed%d"%(seed)+".h5"
        simulation = Simulation.init_from_file(fname)
    data_collector = DataCollector(simulation)
    
    return simulation.lattice


"""

### CLUSTERING ###

"""

# Define a Custom Distance Function including periodic boundary conditions for the x direction 
def periodic_distance_x(s1, s2, width): 
    dx = np.abs(s1[1] - s2[1])
    dy = np.abs(s1[0] - s2[0])
    
    # Apply periodic boundary conditions only along the x direction
    if dx > width / 2:
        dx = width - dx
    
    return dx + dy #np.sqrt(dx**2 + dy**2)


# Calculate the Distance Matrix
def calculate_distance_matrix_x(points, width):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = periodic_distance_x(points[i], points[j], width)
            distance_matrix[j, i] = distance_matrix[i, j]
    
    return distance_matrix

def detect_clusters(data, dendrogram_cutoff, method='single'):
    # Assume already binarized; find directly occupied coordinates
    #occupied_pts = np.transpose(np.nonzero(data))
    occupied_pts = (np.nonzero(data))

    print(np.shape(occupied_pts))
    dim = np.shape(occupied_pts)[1]
    if dim == 1:
        X = occupied_pts[:,0]
        Y = np.zeros(X.shape)
    elif dim == 2:
        X = occupied_pts[:,1]
        Y = occupied_pts[:,0]
    else:
        raise NotImplementedError("Function designed to handle only 1D/2D data. Higher\
        dimensions are not yet allowed.")

    # Calculate the full distance matrix
    distance_matrix = calculate_distance_matrix_x(occupied_pts, width)

    # Convert to condensed distance matrix
    condensed_distance_matrix = distance_matrix[np.triu_indices(len(occupied_pts), k=1)]

    
    # Generate linkage matrix, which describes the iterative clustering steps
    # And then, identifiy each point with a cluster
    linkage_matrix = linkage(condensed_distance_matrix, method=method)
    clusters = fcluster(linkage_matrix, dendrogram_cutoff, criterion="distance")
    clusters = clusters - 1. # Broadcasted substraction; change to zero-base index

    # Engineering the feature matrix
    # Columns: (X coordinate, Y coordinate, cluster id)
    # Casting to integer; saves space with no loss of information
    HCStats = np.stack((X,Y,clusters), axis = 1)
    return linkage_matrix, HCStats.astype(np.int32)



def cluster_statistics(data): 
    ''' Perform hierarchical clustering '''
    linkage_matrix, HCStats = detect_clusters(data = data, method='single', dendrogram_cutoff=cutoff)

    data_mX = HCStats[:,0]
    data_mY = HCStats[:,1]
    cluster_ids = HCStats[:,2]
    unique_cluster_ids, sizes = np.unique(HCStats[:,-1], return_counts=True)
    #print('total number of clusters = %d \n' %(len(unique_cluster_ids)))
    #print('average size of clusters = %g std = %g \n' %(np.mean(sizes), np.mean(sizes)))
    
    mask_less1 = (sizes != 1)
    mask_less123 = (sizes > 3)
    mask_lessmax = (sizes != max(sizes))
    
    unique_cluster_ids_less1 = unique_cluster_ids * mask_less1
    unique_cluster_ids_less123 = unique_cluster_ids * mask_less123
    unique_cluster_ids_filtered1 = [element for element in unique_cluster_ids_less1 if element != 0]
    unique_cluster_ids_filtered123 = [element for element in unique_cluster_ids_less123 if element != 0]
    
    return len(unique_cluster_ids), np.mean(sizes), np.std(sizes), max(sizes), np.std(sizes*mask_lessmax), len(unique_cluster_ids_filtered1), len(unique_cluster_ids_filtered123)


cutoff = 1.1

#example for statistics on clusters:
num_clusters_t = [] #total number of clusters over time
mean_size_t = [] #mean size of clusters in the lattice over time
std_size_t = [] #standard deviation of the clusters size over time
max_size_t = [] #maximum cluster size over time
restricted_std_size_t = [] #standard deviation after removing the maximum size
num_clusters_less_size1_t = [] #total number of clusters (except for those of size 1) over time
num_clusters_less_size123_t = [] #total number of clusters (except for those of size 1,2 or 3) over time

for t in range(50,500,50):
    lattice = read_lattice(t0 = t, tmax = 500, g = 0.4, v0 = 100., v1 = 20.)
    data = torch.sum(lattice.particles, axis = 0)
    cutoff = 1.1 # Cut to dendrogram to obtain clusters at desired scale

    num_clusters , mean_size, std_size, max_size, restricted_std_size,num_clusters_less_size1,num_clusters_less_size123 = cluster_statistics(data)
    num_clusters_t.append(num_clusters)
    mean_size_t.append(mean_size)
    std_size_t.append(std_size)
    max_size_t.append(max_size)
    restricted_std_size_t.append(restricted_std_size)
    num_clusters_less_size1_t.append(num_clusters_less_size1)
    num_clusters_less_size123_t.append(num_clusters_less_size123)



