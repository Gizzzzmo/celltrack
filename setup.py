import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# width and height of the images, will be scaled automatically
width = 2**8
height = 2**8

# the z coordinate of the plane in which the simulated cells will be placed with pyredner
depth = 507

# the number of vertices used to simuale one cell
granularity = 100


# creating a coordinate grid to apply functions to
xx = torch.arange(0, width, 1, device=device, dtype=torch.float32)
yy = torch.arange(0, height, 1, device=device, dtype=torch.float32)
xxx = xx.expand((height, -1))
yyy = yy.expand((width, -1))
yyy = yyy.transpose(0, 1)

xy = torch.stack([xxx, yyy], dim=1)