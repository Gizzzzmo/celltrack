import load
import matplotlib.pyplot as plt
import cell
import imageio
import torch
import glob
from setup import pyredner
from cell import render_vertex_list, render_simulation, redner_simulation

a = load.simulated_ellipses()
cells = a[49]

for c in cells:
    c.create_polygon(154)

closed = []
for path in sorted(glob.glob('../data/stemcells/closed01/*.png')):
    closed.append(0.5*torch.from_numpy(imageio.imread(path)).float())

original = []

for path in sorted(glob.glob('../data/stemcells/01/*.tif')):
    original.append(torch.from_numpy(imageio.imread(path)).float())

def show(i):
    for c in a[i]:
        c.create_polygon(154)
    sim = render_vertex_list(a[i], 255, original[i+1])
    sim0 = render_vertex_list(a[i], 255, closed[i+1])

    plt.figure(1)
    plt.imshow(sim.detach())
    plt.figure(2)
    plt.imshow(sim0.detach())

    if(pyredner is not None):
        for c in a[i]:
            c.create_shape(154)
        sim1 = redner_simulation(a[i])
        plt.figure(3)
        plt.imshow(sim1.detach())
    plt.show()