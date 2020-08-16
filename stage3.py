from setup import *
import pyredner
import torch
import glob
from sklearn.linear_model import LogisticRegression
import imageio
import numpy as np
from PIL import Image
from functools import reduce
from sortedcontainers import SortedList
from load import simulated_cell_vertices_plus_reflectances, simulated_ellipses
from scene import create_scene, create_scene_environment, wiggled_gradients
import matplotlib.pyplot as plt

def run_stage(circles, alpha_lambda):
    generated_cells_directory = '../data/stemcells/simulated/other_initilization_'+str(circles)+'x'+str(circles)+'circles_20/optimized_vertex_lists_1.0_0.01_'+str(alpha_lambda)
    print(generated_cells_directory)
    originals = [[]] + sorted(glob.glob('../data/stemcells/01/*.tif'))
    originals = reduce(lambda a, b: a + [imageio.imread(b)], originals)

    gts = [[]] + sorted(glob.glob('../data/stemcells/01_GT/TRA/man_track???.tif'))
    gts = reduce(lambda a, b: a + [imageio.imread(b)], gts)
    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    scenes = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=1)
    print(len(gts), len(scenes))
    reflectances = []
    labels = []
    cell_images = []

    for cells, gt, orig in zip(scenes[::5], gts[::5], originals[::5]):
        if plotit:
            shapes, render = create_scene(pyredner, cam, shape_light, area_lights, cells)
            img = render(4, 1).sum(dim=-1)
            plt.figure(1)
            plt.imshow(img.cpu().detach())
            plt.figure(2)
            plt.imshow(orig)
            plt.figure(3)
            plt.imshow(gt)
            plt.show()
        for cell in cells:

            shapes, render = create_scene(pyredner, cam, shape_light, area_lights, [cell])
            img = render(4, 1).sum(dim=-1)
            resized_img = np.array(Image.fromarray(img.cpu().detach().numpy()).resize((1024, 1024)))

            weight = (img != 0).sum().float()
            if weight == 0:
                continue

            weighted_center = ((torch.stack([img], dim=2) != 0).float() * xy.permute(0, 2, 1)).sum(dim=(0, 1)) * 1024/(img.shape[0] * weight)

            feature_window_length = alpha * weight.sqrt() * 1024/img.shape[0]

            x_left = int(weighted_center[1] - feature_window_length/2)
            x_right = int(weighted_center[1] + feature_window_length/2)
            y_left = int(weighted_center[0] - feature_window_length/2)
            y_right = int(weighted_center[0] + feature_window_length/2)

            feauture_window = resized_img[x_left:x_right, y_left:y_right]
            aaa = gt[x_left:x_right, y_left:y_right]
            bbb = np.array(Image.fromarray(orig[x_left:x_right, y_left:y_right]).resize((24, 24)))
            bbb = bbb/np.sum(bbb)
            if np.sum(np.isnan(bbb) > 0):
                continue
            reflectances.append(cell.diffuse_reflectance[0].item())
            cell_images.append(bbb)
            if plotit:
                plt.figure(1)
                plt.imshow(feauture_window)
                plt.figure(2)
                plt.imshow(bbb)
                plt.figure(3)
                plt.imshow(aaa)
                plt.show()

            pred = gt[resized_img != 0]
            mostoverlap = 0
            overlap = 0
            print(np.unique(pred))
            for i in np.unique(pred):
                if i != 0 and np.count_nonzero(pred == i) > overlap:
                    mostoverlap = i
                    overlap = np.count_nonzero(pred == i)
                    print(mostoverlap, overlap)
            if mostoverlap == 0 or np.count_nonzero(pred == 0) > 4*overlap:
                labels.append(0)
                print('no')
            else:
                print('yes')
                labels.append(overlap/np.count_nonzero(gt == mostoverlap))
            print(len(labels), len(reflectances))
            np.count_nonzero(gt == mostoverlap)

    X = np.array(reflectances).reshape(len(reflectances), 1)
    y = np.array(labels) > 0
    print(len(X), len(y))
    print(np.sum(np.isnan(cell_images)))
    for i, cell_img in enumerate(cell_images):
        if np.sum(np.isnan(cell_img)) > 0:
            print(i, np.sum(np.isnan(cell_img)))

    np.save(generated_cells_directory + '/cellimages.npy', np.stack(cell_images))
    np.save(generated_cells_directory + '/refl.npy', X)
    np.save(generated_cells_directory + '/labels.npy', y)


plotit = False

for circles in range(7, 14):
    for alpha_lambda in [0, 0.5, 1, 5, 10, 50]:
        run_stage(circles, alpha_lambda)