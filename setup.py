import torch
import numpy as np

# device that torch uses to perform computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# width and height of the images, will be scaled automatically
width = 2**8
height = 2**8

# the z coordinate of the plane in which the simulated cells will be placed with pyredner
depth = 507

# the number of vertices used to simuale one cell with pyredner
granularity = 20

# a parameter to determine the size of the image cutouts of simulated cells
alpha = 1


# creating a coordinate grid to apply functions to
xx = torch.arange(0, width, 1, device=device, dtype=torch.float32)
yy = torch.arange(0, height, 1, device=device, dtype=torch.float32)
xxx = xx.expand((height, -1))
yyy = yy.expand((width, -1))
yyy = yyy.transpose(0, 1)

xy = torch.stack([xxx, yyy], dim=1)


def density_diagram(data):
    """ generate a kernel density estimation"""
    min_value = min(data)
    max_value = max(data)
    x_values = list(range(100))
    y_values = np.zeros((len(x_values)))
    for i in x_values:
        x = min_value + i * (max_value-min_value)/100
        x_values[i] = x
        for grad in data:
            y_values[i] += np.exp(-np.power(grad-x, 2)/((max_value-min_value)/30)**2)
    return x_values, y_values