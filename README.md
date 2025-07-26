# Variational Monte Carlo for Helium-4 Ground State

## Project Overview

This project aims to compute the ground state energy of the Helium-4 atom  using the Variational Monte Carlo method.  
We test several trial wave functions, analyze and plot the results, and a visualization of the configuration space sampling is performed for the best parameters.

---

 Repository Structure

- **`hevmc.py`**  
Performs a scan over different trial wave function parameters and calculates the corresponding ground state energy for each set, using 20'000 samples for each parameter combination.
  Results are saved in `result.txt`.

- **`result.txt`**  
  Output data: Each line contains the mean energy, standard error, trial parameters (alpha, beta, gamma), acceptance ratio, and step size.

- **`plot_energy.py`**  
  Reads the results in `result.txt` and produces 3D plots to visualize the energy landscape, highlight the minimum energy point.

- **`best_wave.py`**  
  Runs a VMC simulation using the "best" trial wave function found previously.  
  Plots the sampled space configuration of the system to check the consistency and quality of the algorithm.



## Python Version

- The code was developed and tested with **Python 3.10.12**.



## Required Libraries

- `numpy`
- `matplotlib`







