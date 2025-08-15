import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from gudhi import AlphaComplex

def compute_alpha_number(coordinates):
    alpha_complex = AlphaComplex(coordinates)
    simplex_tree = alpha_complex.create_simplex_tree()
    persistence = simplex_tree.persistence()
    betti0_deaths = [death for dim, (birth, death) in persistence if dim == 0 and death != float('inf')]
    alpha_number = max(betti0_deaths) if betti0_deaths else 0
    return alpha_number

def plot_voronoi(coordinates, output_path='voronoi_diagram.png'):
    vor = Voronoi(coordinates)
    fig, ax = plt.subplots(figsize=(10, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=20, c='red', label='Nuclei')
    plt.title('Voronoi Diagram of Nuclei')
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')
    plt.legend()
    plt.savefig(output_path) #youll find it in the directory
    plt.close()

def plot_predictions(coordinates, predictions, output_path='gcn_predictions.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=predictions, cmap='viridis', s=20)
    plt.colorbar(label='Predicted Surroundings (0: Unfused, 1: Syncytial-Dominated)')
    plt.title('Predicted Surroundings of Nuclei')
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')
    plt.savefig(output_path)
    plt.close()