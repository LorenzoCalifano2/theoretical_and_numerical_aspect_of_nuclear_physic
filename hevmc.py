from math import exp, sqrt
from random import random, seed
import numpy as np
import sys
#in this code we will calculate the energy of the sysytem using the Monte Carlo method by considering 11 x 11 x11 combinations of the parameters alpha, beta, gamma in the trial wave function

NA=4   #number of particles of the system
h2_dividedby_2m=20.74
h=0.001  # step size for the numerical derivative
NumberMCcycles=20000 #number of Monte Carlo cycles
Ntherms=10 #number of thermalization steps
MaxVariation=11 # variation of the parameters alpha, beta, gamma



def wave_function(R, alpha, beta, gamma):  #trial wave function
    wf = 1.0
    for i in range(NA - 1):
        for j in range(i + 1, NA):
            r2 = np.sum((R[i] - R[j])**2)  
            wf *= exp(gamma* r2) + alpha * exp(beta * r2)
    return wf


def potential(R): 
    epot = 0.0
    r=0
    for i in range(NA - 1):
        for j in range(i + 1, NA):
            r = np.sum((R[i] - R[j])**2)
            epot += 1000.0 * exp(-3.0 * r) - 163.35 * exp(-1.05 * r) - 83.0 * exp(-0.8 * r) - 21.5 * exp(-0.6 * r) - 11.5 * exp(-0.4 * r)

    return epot

def kinetic_energy(wavef, R, alpha, beta, gamma):
    ekinetic = 0.0
    RP = np.copy(R)
    RM = np.copy(R)
#calculate the numerically the laplacian of the wave function
    for i in range(NA):
        for k in range(3):
            RP[i,k] += h
            RM[i,k] -= h
            wf_plus = wave_function(RP, alpha, beta, gamma)
            wf_minus = wave_function(RM, alpha, beta, gamma)
            ekinetic += (wf_plus + wf_minus - 2 * wavef) / h**2 # Second derivative (Laplacian)
            # Restore positions
            RP[i,k] = R[i,k]   
            RM[i,k] = R[i,k]  

    ekinetic *= -h2_dividedby_2m / wavef
    return ekinetic

def local_energy(wavef, R, alpha, beta, gamma):
    ekinetic = kinetic_energy(wavef, R, alpha, beta, gamma)
    epotential = potential(R)
    return ekinetic + epotential


def MonteCarloSampling():
    Energies = np.zeros((MaxVariation, MaxVariation, MaxVariation), np.double) #create energies vector for all the possible combinations of alpha, beta, gamma
    Errors = np.zeros((MaxVariation, MaxVariation, MaxVariation), np.double) #create errors vector for all the possible combinations of alpha, beta, gamma
    Acceptancies= np.zeros((MaxVariation, MaxVariation, MaxVariation), np.double) #create acceptancies vector for all the possible combinations of alpha, beta, gamma
    AlphaValues = np.zeros(MaxVariation, np.double)
    BetaValues = np.zeros(MaxVariation, np.double)
    gammaValues = np.zeros(MaxVariation, np.double)
    filename = f"resul_{MaxVariation}_{NumberMCcycles}.txt"  #creation of the file where the results will be saved
    outfile = open(filename, "w")
    outfile.write("mean_energy error alpha beta gamma acceptance stepsize\n") 
    seed(1234)
    alpha = -0.7191 - ((MaxVariation - 1)/2) * 0.5 # values for alpha centered around -0.7191 and spaced by 0.5
    for ia in range(MaxVariation):
        alpha +=0.5 
        AlphaValues[ia] = alpha
        beta = -2.13796 - ((MaxVariation - 1)/2) * 0.35 # values for beta centered around -2.13796 and spaced by 0.35
        for jb in range(MaxVariation):
            beta += 0.35  
            BetaValues[jb] = beta
            gamma = -0.08597 - ((MaxVariation - 1)/2) * 0.012 # values for gamma centered around -0.08597 and spaced by 0.012
            for kc in range(MaxVariation):
                gamma += 0.012  
                gammaValues[kc] = gamma
                X_Old = 10 * ((np.random.rand(NA, 3) - 0.5))  # (NA,3) matrix of random numbers between -0.5 and 0.5
                wave_Old = wave_function(X_Old, alpha, beta, gamma)
                energy = 0 # Initialize energy
                energy2 = 0 # Initialize energy squared
                Naccepted = 0 # Initialize accepted moves counter
                StepSize=2 #step size for the Monte Carlo sampling
                MCcycle = 0
                while MCcycle < NumberMCcycles: #let's do the Monte Carlo sampling
                    for _ in range(Ntherms): # Thermalization steps
                        for i in range(NA):
                            X_new = np.copy(X_Old)
                            X_new[i] += StepSize * (np.random.rand(3) - 0.5)
                            wave_New = wave_function(X_new, alpha, beta, gamma)
                            if abs(wave_Old) < 1e-250 or abs(wave_Old) > 1e250:# Abort if wave function is numerically unstable
                                print("wave_Old:", wave_Old, "alpha:", alpha, "beta:", beta, "gamma:", gamma)
                                sys.exit("non_valid wave_old: exit") 

                            if abs(wave_New) < 1e-250 or abs(wave_New) > 1e250: # Abort if wave function is numerically unstable
                                print("wave_New:", wave_New, "alpha:", alpha, "beta:", beta, "gamma:", gamma)
                                sys.exit("non_valid wave_New: exit") 
                            else:
                                log_ratio = 2 * (np.log(abs(wave_New)) - np.log(abs(wave_Old))) #Metropoli algorithm
                                if log_ratio >= np.log(np.random.rand()):
                                    X_Old[i] = X_new[i]
                                    wave_Old = wave_New
                                    Naccepted += 1
                    elocal = local_energy(wave_Old, X_Old, alpha, beta, gamma)
                    energy += elocal
                    energy2 += elocal ** 2
                    if MCcycle == NumberMCcycles // 2: # Check the acceptance rate at mid-point
                        acceptance_mid = Naccepted / (MCcycle * Ntherms * NA)
                        if acceptance_mid > 0.8:  # if acceptance rate is too high, increase step size and restart
                            StepSize += 0.5
                            MCcycle = 0
                            Naccepted = 0
                            energy = 0
                            energy2 = 0
                            X_Old = 10 * ((np.random.rand(NA, 3) - 0.5))
                            wave_Old = wave_function(X_Old, alpha, beta, gamma)
                            print("StepSize increased to", StepSize)
                            continue
                        elif acceptance_mid < 0.4: # if acceptance rate is too low, decrease step size and restart
                            StepSize -= 0.5
                            MCcycle = 0
                            Naccepted = 0
                            energy = 0
                            energy2 = 0
                            X_Old = 10 * ((np.random.rand(NA, 3) - 0.5))
                            wave_Old = wave_function(X_Old, alpha, beta, gamma)
                            print("StepSize decreased to", StepSize)
                            continue
                    MCcycle += 1
                mean_energy = energy / NumberMCcycles
                mean_energy2 = energy2 / NumberMCcycles
                error = sqrt((mean_energy2 - mean_energy ** 2) / NumberMCcycles)
                acceptance = 100.0 * Naccepted / (NumberMCcycles * Ntherms * NA)
                Energies[ia, jb, kc] = mean_energy
                Errors[ia, jb, kc] = error
                Acceptancies[ia, jb, kc] = acceptance


                
                


                outfile.write('%10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n' % (mean_energy, error, alpha, beta, gamma, acceptance, StepSize)) #write the values in the file the energy, error, alpha, beta, gamma, acceptance and step size

    outfile.close()

    return Energies, Errors, AlphaValues, BetaValues, gammaValues, Acceptancies



Energies, Errors, AlphaValues, BetaValues, gammaValues, Acceptancies = MonteCarloSampling()


min_index = np.unravel_index(np.argmin(Energies), Energies.shape) #find the index of the minimum energy value
ia, jb, kc = min_index

min_energy = Energies[ia, jb, kc]
min_error = Errors[ia, jb, kc]
min_alpha = AlphaValues[ia]
min_beta = BetaValues[jb]
min_gamma = gammaValues[kc]

print(f"Energia minima: {min_energy:.6f} ± {min_error:.6f}")
print(f"alpha = {min_alpha}, beta = {min_beta}, gamma = {min_gamma}")


