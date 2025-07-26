import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# this file just reads energy data from the file "results.txt", generates a single 3D plot, highlights the point with minimum energy in red
    

def plot_energy_3d_min_highlight(filename="result.txt"):
    mean_energy = []
    error_energy = []
    alpha = []
    beta = []
    gamma = []

    try:
        with open(filename, 'r') as f:

            next(f)    # Skip the header line, which contains column names
            for line in f: # Read each line in the file
                parts = line.strip().split(',')   # if the file uses commas
                #parts = line.strip().split()       # if the file uses spaces instead of commas
                if len(parts) >= 5:
                    mean_energy.append(float(parts[0]))
                    error_energy.append(float(parts[1]))
                    alpha.append(float(parts[2]))
                    beta.append(float(parts[3]))
                    gamma.append(float(parts[4]))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

  # Convert lists to numpy arrays for convenienc
    mean_energy = np.array(mean_energy)
    error_energy = np.array(error_energy)
    alpha = np.array(alpha)
    beta = np.array(beta)
    gamma = np.array(gamma)

    # Find index and values of the minimum energy point
    min_energy_idx = np.argmin(mean_energy)
    min_energy_value = mean_energy[min_energy_idx]
    min_error_value = error_energy[min_energy_idx]
    min_alpha = alpha[min_energy_idx]
    min_beta = beta[min_energy_idx]
    min_gamma = gamma[min_energy_idx]

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for all points, with color representing mean_energy
    # Set limits for the colorbar and normalization for sensitivity between -24.7 and -20
    from matplotlib import colors
    vmin = -24.7
    vmax = 2056

    shift = abs(vmin) + 1 # shift to avoid negative values in the log
    norm = colors.LogNorm(vmin=vmin+shift, vmax=vmax+shift)
    sc = ax.scatter(alpha, beta, gamma, c=mean_energy+shift, cmap='viridis', s=50, norm=norm)

    # Highlight the minimum energy point in red
    # Updated label to include alpha, beta, and gamma values
    label_min_energy = (
        f'E_min = {min_energy_value:.3f} ± {min_error_value:.3f}\n'
        f'($\\alpha$={min_alpha:.3f}, $\\beta$={min_beta:.3f}, $\\gamma$={min_gamma:.3f})'
    )
    ax.scatter(min_alpha, min_beta, min_gamma, color='red', marker='o', s=200, label=label_min_energy)

    # Set labels
    ax.set_xlabel('Alpha ($\\alpha$)')
    ax.set_ylabel('Beta ($\\beta$)')
    ax.set_zlabel('Gamma ($\\gamma$)')
    ax.set_title('Mean Energy in ($\\alpha$, $\\beta$, $\\gamma$)')

    # Add a color bar to show the energy scale
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Mean Energy')

    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.show()

# Run the plotting function
plot_energy_3d_min_highlight("result.txt")
