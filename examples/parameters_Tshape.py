import torch
h = 6
lb = int(1.5*h)
lc = 4*h

# Parameters for ParticleLattice
width = h+lc+2
height = h+2*lb+2

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 100.0  # Base transition rate

obstacles = torch.zeros((height, width), dtype=torch.bool)
obstacles[0:(lb+1),0:(lc+1)] = True
obstacles[(lb+h+1):(h+2*lb+2),0:(lc+1)] = True
obstacles[:, -1] = True

sinks = torch.zeros((height, width), dtype=torch.bool)
sinks[-1,(lc+1):(lc+h+1)] = True
sinks[0,(lc+1):(lc+h+1)]  = True
sinks[(lb+1):(lb+h+1),0]  = True

X = []
Y = []
Y1 = list(range(lb+1,lb+h+1,1))
Y2 = Y1
for i in range(12):
    X = X+[2*i+1]*len(Y1)+[2*i+2]*len(Y2)
    Y = Y+Y1+Y2
list_part_right = [X, Y]