import torch
from vlmc.core.particle_lattice import ParticleLattice
from vlmc.core.magnetic_field import MagneticField
from vlmc.core.simulation import Simulation
from tqdm import tqdm
from examples.utils import *
import numpy as np
import random

#chiara: added seed
seed = 3179
phi = 0
torch.manual_seed(seed)
np.random.seed(seed)


def main():
    # Parameters for ParticleLattice
    height, width = 25, 51
    obstacle_topology, sink_topology = "none", "none" 
    obstacle_walls, sink_walls = ["top", "bottom"], "none" #["left","right"]
    obstacles, sinks = generate_lattice_topology(
        height, width, obstacle_topology, sink_topology, obstacle_walls, sink_walls
    )    

    lattice_params = {
        # include relevant parameters here
        "width": width,
        "height": height,
        "density": 0.3,
        "obstacles": obstacles,
        "sinks": sinks,
    }

    # Creating a ParticleLattice instance
    Lattice = ParticleLattice(**lattice_params) #CHIARA: the lattice is initialized in SImulation? I modified that. 
    #Lattice = 0
    print ('obstacles  == ' , Lattice.obstacles.sum().sum())
    print ('obstacles  == ' , obstacles.sum().sum())
    
    # Creating a MagneticField instance
    magnetic_field = MagneticField()  # Add parameters if needed

    # Simulation parameters
    g = 1.0  # Alignment sensitivity
    v0 = 100.0  # Base transition rate
    v1 = 0.0 # flow rate 
    magnetic_field_interval = 0.02  # Time interval for magnetic field application

    # Initialize the Simulation
    simulation = Simulation(Lattice, g, v0, v1, magnetic_field_interval, **lattice_params)

    N0 = simulation.lattice.particles[:].sum().item() #initial number of particles
    
    #simulation = Simulation(g, v0, magnetic_field_interval, **lattice_params)

    t_last = 0
    n_steps = int(1.e5) # Number of steps to run the simulation for
    order_params = np.zeros(n_steps)
    
    print("Initial lattice")
    #print(simulation.lattice)
    print(Lattice)
    print(f"Initial order parameter: {simulation.lattice.compute_order_parameter()}")
    # create a mapping between the event codes and the event names for printing purposes
    # 0 is up 1 is left 2 is down 3 is right 4 is migrate
    event_names = [
        "turned up",
        "turned left",
        "turned down",
        "turned right",
        "migrated to the next cell",
    ]
    event_codes = [0, 1, 2, 3, 4]
    event_map = dict(zip(event_codes, event_names))

    fileObject = open(f"data_withheuristic/order_params_size{height}{width}_seed{seed}_v0{v0}_v1{v1}_g{g}.dat", "w")
    fileDensity_y = open(f"data_withheuristic/density_y_size{height}{width}_seed{seed}_v0{v0}_v1{v1}_g{g}.dat", "w")
    fileVely_x = open(f"data_withheuristic/meanvely_x_size{height}{width}_seed{seed}_v0{v0}_v1{v1}_g{g}.dat", "w")
    fileVely_y = open(f"data_withheuristic/meanvely_y_size{height}{width}_seed{seed}_v0{v0}_v1{v1}_g{g}.dat", "w")
    file_particles_in_sinks = open(f"data_withheuristic/numparticles_in_sinks{height}{width}_seed{seed}_v0{v0}_v1{v1}_g{g}.dat", "w")
    file_lattice = open(f"data_withheuristic/lattice_saved_{height}{width}_seed{seed}_v0{v0}_v1{v1}_g{g}.dat", "w")



    print ('obstacles == ' , simulation.lattice.obstacles.sum().sum())
    for _ in tqdm(range(n_steps)):                
        
        if _ %5e3==0: #save quantities
            array_to_save = np.array([_, simulation.lattice.particles[:].sum().sum(), simulation.lattice.count_region[0],simulation.lattice.count_region[1], simulation.lattice.count_region[2], simulation.lattice.count_region[3], simulation.lattice.count_region[4]])
            np.savetxt(file_particles_in_sinks, array_to_save.reshape(1,7), fmt='%d', delimiter = ' ')            
            file_particles_in_sinks.flush()

        if _ %100==0: #save quantities 
            for idx_reg in range(5):
                simulation.lattice.count_region[idx_reg]=0                
            #simulation.lattice.griglia --  save lattice
            lattice_to_array = []
            for ev in range(len(event_names)-1):
                for j in range (height):
                    for i in range (width):
                        lattice_to_array.append(simulation.lattice.particles[ev,j,i])
            np.savetxt(file_lattice, np.array(lattice_to_array).reshape(1,len(lattice_to_array)), fmt='%d', delimiter = ' ')

            # ev*(height*width) + y*height + x 
            
        if _ %20==0:
            print(simulation.lattice)
            order_params[_] = simulation.lattice.compute_order_parameter()
            array_to_save = np.array([_,simulation.t, order_params[_]]) 
            np.savetxt(fileObject, array_to_save.reshape(1,3), fmt='%g', delimiter = ' ')
            fileObject.flush()
            
            density_y = simulation.lattice.compute_density_as_function_of_y()
            arr_y = [_, simulation.t]
            for j in range(len(density_y)):
                arr_y.append(density_y[j])
            array_to_save = np.array(arr_y)
            np.savetxt(fileDensity_y, array_to_save.reshape(1,len(density_y)+2), fmt='%g', delimiter = ' ')
            fileDensity_y.flush()

            mean_vel_y = simulation.lattice.compute_meanvelocity_as_function_of_y()
            array_to_savex = np.array(mean_vel_y[0])
            array_to_savey = np.array(mean_vel_y[1])
            np.savetxt(fileVely_x, array_to_savex.reshape(1,len(mean_vel_y[0])), fmt='%g', delimiter = ' ')
            fileVely_x.flush()
            np.savetxt(fileVely_y, array_to_savey.reshape(1,len(mean_vel_y[1])), fmt='%g', delimiter = ' ')
            fileVely_y.flush()


            

        if simulation.lattice.particles[:].sum().item() > 0:

            if simulation.test_apply_magnetic_field():
                if _ >= 2.e4:
                #if simulation.t>10.:
                    n1,vx1,vy1 = simulation.lattice.row_average(1,int((height-1)/3.))
                    vy1=-vy1
                    n2,vx2,vy2 = simulation.lattice.row_average(int((height-1)/3.),2*int((height-1)/3.)+1)
                    vy2=-vy2
                    n3,vx3,vy3 = simulation.lattice.row_average(2*int((height-1)/3.)+1,height-1)
                    vy3=-vy3
                    if n1>n2 and n1>n3:
                        if vy1<-abs(vx1):
                            direction = 0
                        else:
                            if vx1>0:
                                direction = -1
                            else:
                                direction = +1

                    elif n2>n1 and n2>n3:
                        if vx2<0 and vy2>0:
                            direction = +1
                        elif vx2<0 and vy2<0:
                            direction = -1
                        else:
                            direction = 0
                        
                    else:
                        if vy3>abs(vx3):
                            direction = 0
                        else:
                            if vx3>0:
                                direction = +1
                            else:
                                direction = -1

                                
                
                else:
                    direction = 0
                    n1=0
                    n2=0
                    n3=0
                    vx1=0
                    vy1=0
                    vx2=0
                    vy2=0
                    vx3=0
                    vy3=0

                print('Applying magnetic field,time=%g, n1=%d, n2=%d, n3=%d and direction = %d'%(simulation.t,n1,n2,n3,direction))
                print('velocity, time=%g, v1x=%g, v1y=%g, v2x=%g, v2y=%g, v3x=%g, v3y=%g'%(simulation.t,vx1,vy1,vx2,vy2,vx3,vy3))
                simulation.magnetic_field.set_direction(direction)                    
                
            event = simulation.run()
            dt = simulation.t - t_last
            t_last = simulation.t
            scra = np.random.uniform(0,1)
            
        else:
            scra = 1
            dt = torch.distributions.Exponential(phi).sample().item()
            simulation.t = t_last + dt
            t_last = simulation.t 

        if simulation.lattice.particles[:].sum().item() < N0:
        #if scra < phi*dt: #for flux added independent of number of particles in the lattice. 
            # define the region
            #if squared lattice
            region = [[2,width-2,1,2], [2,width-2,height-2,height-1], [1,2,2,height-2], [width-2,width-1,2,height-2]]
            # # region = [[1,2,1,height-1]]
            number_added = 0
            #selected_index = np.random.randint(0,4) for square lattice
            selected_index = 2
            possible_orientation = ["down", "up", "right", "left"]                

            for rr in range(1): #(0,4) square lattice
                #orient = possible_orientation[(selected_index+rr)%4]
                orient = possible_orientation[np.random.randint(0,4)]
                number_added = simulation.lattice.add_particle_flux(1,region[(selected_index+rr)%4],orient)
                if number_added != 0:
                    simulation.update_rates()
                    print('added_particlex_flux at region %d with orientation %s' %(selected_index,orient))
                    break
                
            if number_added == 0:
                print('unable to add the particle')
                break
        
                
        
    #print("Final lattice")
    #print(simulation.lattice.griglia)
    print(f"Final order parameter: {simulation.lattice.compute_order_parameter()}")

if __name__ == "__main__":
    main()
