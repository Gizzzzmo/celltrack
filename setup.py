import torch
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# width and height of the images, will be scaled automatically
width = 2**8
height = 2**8

# the z coordinate of the plane in which the simulated cells will be placed with pyredner
depth = 507

# the number of vertices used to simuale one cell
granularity = 20

alpha = 0.7


# creating a coordinate grid to apply functions to
xx = torch.arange(0, width, 1, device=device, dtype=torch.float32)
yy = torch.arange(0, height, 1, device=device, dtype=torch.float32)
xxx = xx.expand((height, -1))
yyy = yy.expand((width, -1))
yyy = yyy.transpose(0, 1)

xy = torch.stack([xxx, yyy], dim=1)

def density_diagram(gradients):
    min_value = min(gradients)
    max_value = max(gradients)
    x_values = list(range(100))
    y_values = np.zeros((len(x_values)))
    for i in x_values:
        x = min_value + i * (max_value-min_value)/100
        x_values[i] = x
        for grad in gradients:
            y_values[i] += np.exp(-np.power(grad-x, 2)/((max_value-min_value)/30)**2)
    return x_values, y_values

def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array