import numpy as np
import matplotlib.pyplot as plt
import sys
#In the first part of the code we will calculate the energy of the system using Monte Carlo method fot the "best wave function"
#in the second part we will plot a 3D and 2D plot of the distances between particles
NA=4 #number of particles in the system
h2div2m=20.74
h=0.001  # step size for the numerical derivative
step_size=2 # step size for the Monte Carlo moves
Nmoves=8000 # number of Monte Carlo moves
Ntherm=10   # number of thermalization steps

gamma= -0.08597
alpha=-0.7191
beta=-2.13796
def f(r2):
    return np.exp((gamma) * r2) + (alpha) * np.exp((beta) * r2)

def wave_function(R):
    wf = 1.0
    for i in range(NA - 1):
        for j in range(i + 1, NA):
            r2 = np.sum((R[i] - R[j])**2) # (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
            wf *= f(r2)
    return wf


def potential(r2):
    return (1000.0 * np.exp(-3.0 * r2)
            - 163.35 * np.exp(-1.05 * r2)
            - 83.0 * np.exp(-0.8 * r2)
            - 21.5 * np.exp(-0.6 * r2)
            - 11.5 * np.exp(-0.4 * r2))


def elocal(wfold, R):
    ekin = 0.0
    RP = np.copy(R)
    RM = np.copy(R)
#calculate the numerically the laplacian of the wave function
    for i in range(NA):
        for k in range(3):
            RP[i,k] += h
            RM[i,k] -= h
            wf_plus = wave_function(RP)
            wf_minus = wave_function(RM)
            ekin += (wf_plus + wf_minus - 2 * wfold) / h**2
            # Restore positions
            RP[i,k] = R[i,k]
            RM[i,k] = R[i,k]
    ekin *= -h2div2m / wfold
    epot = 0.0
    for i in range(NA - 1):
        for j in range(i + 1, NA):
            r2 = np.sum((R[i] - R[j])**2)
            epot += potential(r2)

    return ekin + epot
positions = [[] for _ in range(NA)]  # List to store positions of each particle usefull for plotting later
def variational_mc():
    R = 10 * step_size * (np.random.rand(NA, 3) - 0.5) # initial position.  It's a (NA,3) matrix of random numbers between -0.5 and 0.5
    wfold =  wave_function(R)
    sume = 0.0
    sume2 = 0.0
    Naccept = 0
    for imove in range(Nmoves): # loop over the number of Monte Carlo moves
        for _ in range(Ntherm): # thermalization steps
            for i in range(NA):
                Rp = np.copy(R)
                Rp[i] += step_size * (np.random.rand(3) - 0.5) #metropolis algorithm
                wfnew = wave_function(Rp)
                if abs(wfold) < 1e-250 or abs(wfold) > 1e250: # Abort if wave function is numerically unstable
                    print("wave_Old:", wfold, "alpha:", alpha, "beta:", beta, "gamma:", gamma)
                    sys.exit("non_valid wave_old: exit")
                if abs(wfnew) < 1e-250 or abs(wfnew) > 1e250: # Abort if wave function is numerically unstable
                                print("wave_New:", wfnew, "alpha:", alpha, "beta:", beta, "gamma:", gamma)
                                sys.exit("non_valid wave_New: exit")
                else:
                    log_ratio = 2 * (np.log(abs(wfnew)) - np.log(abs(wfold)))
                    if log_ratio >= np.log(np.random.rand()):
                        R[i] = Rp[i]
                        wfold = wfnew
                        Naccept += 1

        energy = elocal(wfold, R)
        sume += energy
        sume2 += energy**2

 
        for i in range(NA):
            positions[i].append(np.append(R[i].copy(), imove))  # add the position and time step to the list for plotting later
    e_mean = sume / Nmoves
    e2_mean = sume2 / Nmoves
    error = np.sqrt((e2_mean - e_mean**2) / Nmoves)
    acceptance = 100.0 * Naccept / (Nmoves * Ntherm* NA)
    print(f"Energy   = {e_mean:.5f}")
    print(f"Error    = {error:.5f}")
    print(f"Acceptance percentage = {acceptance:.2f}%")
   #LET'S PLOT THE DISTANCES BETWEEN PARTICLES: we will first plot a 3D plot of the distances between particles, then we will plot a 2D plot of the distances between particles as a function of time
    all_rij_data = [[] for _ in range(NA)] # List of NA lists, each sublist for a particle
    for imove_idx in range(Nmoves):
        current_positions = np.array([pos[imove_idx] for pos in positions]) #array of shape (NA,3). For the current Monte Carlo step (imove_idx), collects the 3D positions of *all* particles
        for i in range(NA):
            ri = current_positions[i] #3D position of the particle i
            rij_for_particle_i = []
            for j in range(NA):
                if i == j:
                    continue
                rj = current_positions[j]
                rij_for_particle_i.append(np.linalg.norm(ri - rj))
            all_rij_data[i].append(rij_for_particle_i)

    # Convert to numpy arrays for easier indexing
    for i in range(NA):
        all_rij_data[i] = np.array(all_rij_data[i]) 
    times = np.arange(Nmoves) # Time steps for plotting
    # --- 3D Plotting ---
    fig = plt.figure(figsize=(16, 12))
    for i in range(NA):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        rij_all_i = all_rij_data[i] # Use the pre-calculated data

        # Prepare axis labels
        axis_labels = [f"r{i+1}{j+1}" for j in range(NA) if j != i]

        scatter = ax.scatter(
            rij_all_i[:, 0], rij_all_i[:, 1], rij_all_i[:, 2],
            c=times, cmap='viridis', marker='o', s=10
        )
        ax.set_title(f'Particle {i+1}')
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
        fig.colorbar(scatter, ax=ax, label='Monte Carlo Step')
    plt.tight_layout()
    plt.show()

    # --- 2D Plotting ---
    for i in range(NA):
        rij_i = all_rij_data[i] # Use the pre-calculated data

        fig, axs = plt.subplots(1, NA-1, figsize=(16, 4))
        labels = [f"r{i+1}{j+1}" for j in range(NA) if j != i]
        for idx in range(NA-1):
            axs[idx].plot(times, rij_i[:, idx], marker='.', linestyle='-', markersize=3)
            axs[idx].set_xlabel('Monte Carlo step')
            axs[idx].set_ylabel(labels[idx])
            axs[idx].set_title(f'{labels[idx]} as a function of time')
            axs[idx].set_ylim(0, 10)
            axs[idx].grid(True)
        fig.suptitle(f'Particle {i+1}: r_ij coordinates as a function of time', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    np.random.seed(1234) 
    variational_mc()
