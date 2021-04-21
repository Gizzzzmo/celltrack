from setup import alpha, xy
import pyredner
import torch
import glob
import imageio
import numpy as np
from PIL import Image
from functools import reduce
from load import simulated_cell_vertices_plus_reflectances
from scene import create_scene, create_scene_environment
import matplotlib.pyplot as plt

def run_stage(circles, alpha_lambda):
    """
        run the stage for the specified number *circles* of starting cells and regularization weight *alpha_lambda*:
        This entails cutting squares out of the target images, where, during the previous stage, a simulated cell ended up,
        and labeling the resulting image image as either containing or not containing an actual cells based on the tracking groun truth.
        All in all the images, the cells' reflectances (for the thresholding) and the resulting labels are saved to be used for training.
    """

    # load the target images
    generated_cells_directory = '../data/stemcells/simulated/other_initilization_'+str(circles)+'x'+str(circles)+'circles_20/optimized_vertex_lists_1.0_0.01_'+str(alpha_lambda)
    print(generated_cells_directory)
    originals = [[]] + sorted(glob.glob('../data/stemcells/01/*.tif'))
    originals = reduce(lambda a, b: a + [imageio.imread(b)], originals)

    # load the tracking ground truths
    gts = [[]] + sorted(glob.glob('../data/stemcells/01_GT/TRA/man_track???.tif'))
    gts = reduce(lambda a, b: a + [imageio.imread(b)], gts)

    # setup a scene environment
    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    # load the scenes resulting from stage1
    scenes = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=1)
    print(len(gts), len(scenes))
    reflectances = []
    labels = []
    cell_images = []

    ik = -1
    # go over all generated scenes, the respective tracking ground truths, and the original target images
    # and cut a square out of the original image whereever there is a cell in the generated scenes.
    # These cutouts are labeled based on whether or not the cell overlaps with a tracking segment from the ground truth
    for cells, gt, orig in zip(scenes[::5], gts[::5], originals[::5]):
        if plotit:
            _, render = create_scene(pyredner, cam, shape_light, area_lights, cells)
            img = render(4, 1).sum(dim=-1)
            plt.figure(1)
            plt.imshow(img.cpu().detach())
            plt.figure(2)
            plt.imshow(orig)
            plt.figure(3)
            plt.imshow(gt)
            plt.show()

        for cell in cells:
            ik += 1

            # render the scene with just the current cell
            _, render = create_scene(pyredner, cam, shape_light, area_lights, [cell])
            img = render(4, 1).sum(dim=-1)
            # resize the image back to 1024 by 1024 (the size of the original images)
            resized_img = np.array(Image.fromarray(img.cpu().detach().numpy()).resize((1024, 1024)))

            # compute how many pixels end up being non zero
            weight = (img != 0).sum().float()
            # if all are zero, skip this cell
            if weight == 0:
                continue

            # compute the weighted center of the current cell
            weighted_center = ((torch.stack([img], dim=2) != 0).float() * xy.permute(0, 2, 1)).sum(dim=(0, 1)) * 1024/(img.shape[0] * weight)

            # compute the size of the square, that gets cutout
            feature_window_length = alpha * weight.sqrt() * 1024/img.shape[0]

            # compute the indices from which to slice to obtain the correct cutout
            x_left = int(weighted_center[1] - feature_window_length/2)
            x_right = int(weighted_center[1] + feature_window_length/2)
            y_left = int(weighted_center[0] - feature_window_length/2)
            y_right = int(weighted_center[0] + feature_window_length/2)

            # cutout the square from the rendering, the ground truth and the original image
            feauture_window = resized_img[x_left:x_right, y_left:y_right]
            aaa = gt[x_left:x_right, y_left:y_right]
            imm = Image.fromarray(orig[x_left:x_right, y_left:y_right]).resize((24, 24))
            bbb = np.array(imm)

            if np.sum(np.isnan(bbb) > 0):
                continue
            # put the reflectance and the cutout into a list
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

            # figure out which tracking segment the rendered has the most overlap with
            pred = gt[resized_img != 0]
            mostoverlap = 0
            overlap = 0
            print(np.unique(pred))
            for i in np.unique(pred):
                if i != 0 and np.count_nonzero(pred == i) > overlap:
                    mostoverlap = i
                    overlap = np.count_nonzero(pred == i)
                    print(mostoverlap, overlap)

            # if no overlap was found or the overlap is less than a fourth of the rendered segment, label as not being a cell
            # otherwise label as being a cell
            if mostoverlap == 0 or np.count_nonzero(pred == 0) > 4*overlap:
                labels.append(0)
                print('no')
            else:
                print('yes')
                labels.append(overlap/np.count_nonzero(gt == mostoverlap))
            print(len(labels), len(reflectances))
            np.count_nonzero(gt == mostoverlap)

    # put all everything into numpy arrays and save the results
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

# toggle to show additional plots during the process
plotit = False

# run stage 2 for variations of starting layouts and regularization weights
for circles in range(7, 14):
    for alpha_lambda in [0, 0.5, 1, 5, 10, 50]:
        run_stage(circles, alpha_lambda)