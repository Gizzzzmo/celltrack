import load
import matplotlib.pyplot as plt
import imageio
import torch
import glob
import pyredner
import numpy as np
from PIL import Image
from cell import render_vertex_list, render_simulation
from scene import create_scene, create_scene_environment
from load import simulated_cell_vertices_plus_reflectances
from functools import reduce

# load simulated ellipses that were the result of the blob optimization
a = load.simulated_ellipses()
cells = a[49]

# for all cells transform their (center, matrix)-representation into a polygon (vertex list) representation
for c in cells:
    c.create_polygon(154)

# load original images
original = []
for path in sorted(glob.glob('../data/stemcells/01/*.tif')):
    original.append(torch.from_numpy(imageio.imread(path)).float())

def show(i):
    """ shows the result of the ith image's *blob* optimization """
    for c in a[i]:
        c.create_polygon(154)
    sim = render_vertex_list(a[i], 255, original[i+1])
    sim1 = render_simulation(a[i])

    plt.figure(1)
    plt.imshow(sim.detach())
    plt.figure(2)
    plt.imshow(sim1.detach())
    plt.show()

def seg_mask(alpha_lambda, circles, i):
    """ produces and saves the ith image out of the first time series overlayed with the generated segmentation mask in yellow"""
    generated_cells_directory = '../data/stemcells/simulated/other_initilization_'+str(circles)+'x'+str(circles)+'circles_20/optimized_vertex_lists_1.0_0.01_'+str(alpha_lambda)

    originals = [[]] + sorted(glob.glob('../data/stemcells/01/*.tif'))
    originals = reduce(lambda a, b: a + [imageio.imread(b)], originals)

    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    scenes = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=1)
    cells = scenes[i]
    orig = originals[i]

    overlay = np.empty((*orig.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = orig
    overlay[:, :, 1] = orig
    overlay[:, :, 2] = orig
    
    for cell in cells:
        cell.diffuse_reflectance = torch.cuda.FloatTensor([100, 100, 100])
        _, render = create_scene(pyredner, cam, shape_light, area_lights, [cell])
        img = render(4, 1).sum(dim=-1)
        img = np.array(Image.fromarray(img.detach().cpu().numpy()).resize((1024, 1024)))
        print(np.sum(img > 50))
        overlay[img > 50, :] //= 6
        overlay[img > 50, :] *= 5
        overlay[img > 50, :] += np.array([241, 192, 0], dtype=np.uint8) //6

    Image.fromarray(overlay, 'RGB').save('results/qualitative/mask.png')
    
    plt.imshow(overlay)
    plt.show()

def seg_mask1(cells, orig, save=None, show=True):
    """ produces and saves a version of the *orig* image overlayed with *cells* in yellow"""
    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    overlay = np.empty((*orig.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = orig
    overlay[:, :, 1] = orig
    overlay[:, :, 2] = orig
    
    for cell in cells:
        last_refl = cell.diffuse_reflectance
        cell.diffuse_reflectance = torch.cuda.FloatTensor([100, 100, 100])
        _, render = create_scene(pyredner, cam, shape_light, area_lights, [cell])
        img = render(4, 1).sum(dim=-1)
        img = np.array(Image.fromarray(img.detach().cpu().numpy()).resize((1024, 1024)))
        print(np.sum(img > 50))
        overlay[img > 50, :] //= 6
        overlay[img > 50, :] *= 5
        overlay[img > 50, :] += np.array([241, 192, 0], dtype=np.uint8) //6
        cell.diffuse_reflectance = last_refl

    if save is not None:
        Image.fromarray(overlay, 'RGB').save('results/qualitative/'+save)
    
    if show:
        plt.imshow(overlay)
        plt.show()
    return overlay