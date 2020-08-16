from setup import *
import pyredner
import torch
import glob
import imageio
import mahotas
import numpy as np
import scipy as sp
from PIL import Image
from functools import reduce
from sortedcontainers import SortedList
from load import simulated_cell_vertices_plus_reflectances, simulated_ellipses
from scene import create_scene, create_scene_environment, wiggled_gradients
import train
import csv
import matplotlib.pyplot as plt


def update_assigned_cell(cell, mask_to_cell_map, classifier_index):
    for jaccard, index in reversed(cell.jaccard[classifier_index]):
        if index not in mask_to_cell_map:
            mask_to_cell_map[index] = cell
            break
        else:
            competing_cell = mask_to_cell_map[index]
            if competing_cell.jaccard[classifier_index][-1][0] < jaccard:
                competing_cell.jaccard[classifier_index].pop()
                mask_to_cell_map[index] = cell
                update_assigned_cell(competing_cell, mask_to_cell_map, classifier_index)
                break
            else:
                cell.jaccard[classifier_index].pop()

def evaluate(scenes, target_masks, originals, *classifiers):

    kernel_size = 7
    sigma = 4

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    blur = torch.nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2)
    blur.weight = torch.nn.Parameter(gaussian_kernel.reshape(1, 1, kernel_size, kernel_size))
    blur.bias[:] = 0
    blur = blur.cuda()

    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    classifiers = list(classifiers)
    classifiers.append(lambda cell: True)

    avg_jaccard = [0] * len(classifiers)
    num_identified_cells = [0] * len(classifiers)
    false_positives = [0] * len(classifiers)
    missed_cells = [0] * len(classifiers)
    num_cells = 0

    test_reflectance = torch.cuda.FloatTensor([100, 100, 100])

    for cells, target_mask, orig in zip(scenes, target_masks, originals):
        print('next')
        mask_to_cell_map = [{} for i in avg_jaccard]
        caught_cells = [0] * len(classifiers)
        for cell in cells:
            #vertices = cell.vertices.cpu().int().numpy()
            #canvas = np.zeros((1024, 1024))
            #mahotas.polygon.fill_polygon(vertices*4, canvas)
            #canvas = canvas.transpose()
            cell.orig = orig
            reflectance = cell.diffuse_reflectance
            cell.diffuse_reflectance = test_reflectance
            _, render = create_scene(pyredner, cam, shape_light, area_lights, [cell])
            cell.img = render(4, 1).sum(dim=-1)
            cell.diffuse_reflectance = reflectance

            resized_img = np.array(Image.fromarray(cell.img.detach().cpu().numpy()).resize((1024, 1024)))
            if plotit:
                print(cell.diffuse_reflectance, classifiers[0](cell))
                plt.figure(1)
                plt.imshow(resized_img)
                plt.figure(2)
                plt.imshow(np.abs(target_mask - (resized_img > 50) * 10))
                #plt.figure(3)
                #plt.imshow(canvas)
                #plt.figure(4)
                #plt.imshow(np.abs(target_mask - canvas * 10))
                plt.show()
            cell.jaccard = [None] * len(classifiers)

            cell.pred = [None] * len(classifiers)
            for i, predict in enumerate(classifiers):
                if predict(cell):
                    cell.jaccard[i] = SortedList(key=lambda t: t[0])
                    cell.pred[i] = target_mask[resized_img  > 50]
                    prediction_size = np.count_nonzero(resized_img > 50)
                    for j in np.unique(cell.pred[i]):
                        if j != 0:
                            overlap = np.count_nonzero(cell.pred[i] == j)
                            reference_size = np.count_nonzero(target_mask == j)
                            union = reference_size + prediction_size - overlap
                            if overlap > 0.5*reference_size:
                                cell.jaccard[i].add((overlap/union, j))

                    update_assigned_cell(cell, mask_to_cell_map[i], i)
            del cell.img

        if plotit:
            _, render = create_scene(pyredner, cam, shape_light, area_lights, cells)
            plt.figure(1)
            plt.imshow(render(4, 1).sum(dim=-1).cpu().detach())
            plt.figure(2)
            plt.imshow(target_mask)
            plt.figure(3)
            plt.imshow(orig)
            plt.show()

        for i in range(len(classifiers)):
            for cell in cells:
                if cell.jaccard[i] is not None:
                    num_identified_cells[i] += 1
                    if not cell.jaccard[i]:
                        false_positives[i] += 1
                    else:
                        caught_cells[i] += 1
                        jaccard, index = cell.jaccard[i][-1]
                        print(jaccard)
                        avg_jaccard[i] += jaccard
        
            missed_cells[i] += len(np.unique(target_mask)) - 1 - caught_cells[i]
        num_cells += len(np.unique(target_mask)) - 1

    stats = []
    for i in range(len(classifiers)):
        avg_false_positives = false_positives[i] / len(scenes)
        avg_missed_cells = missed_cells[i] / len(scenes)

        avg_num_of_cells = num_cells / len(scenes)

        print(avg_jaccard[i])
        if num_cells != 0:
            avg_jaccard[i] /= num_cells

        stats.append((avg_num_of_cells, avg_false_positives, avg_missed_cells, avg_jaccard[i]))

        print('classifier', str(i) + ':', avg_num_of_cells, avg_false_positives, avg_missed_cells, avg_jaccard[i])

    return stats


def predictor(classifier):
    return lambda cell: classifier.predict(cell.diffuse_reflectance[0].detach().cpu().numpy().reshape(1, 1))

def thresholding(threshold):
    return lambda cell: cell.diffuse_reflectance[0].item() > threshold


def crop_predictor(classifier, mean, std):
    def predictor(cell):
        weight = (cell.img != 0).sum().float()
        print(weight)
        if weight == 0:
            return False
        weighted_center = ((torch.stack([cell.img], dim=2) != 0).float() * xy.permute(0, 2, 1)).sum(dim=(0, 1)) * 1024/(cell.img.shape[0] * weight)

        feature_window_length = alpha * weight.sqrt() * 1024/cell.img.shape[0]

        print(weighted_center, feature_window_length)

        x_left = int(weighted_center[1] - feature_window_length/2)
        x_right = int(weighted_center[1] + feature_window_length/2)
        y_left = int(weighted_center[0] - feature_window_length/2)
        y_right = int(weighted_center[0] + feature_window_length/2)

        bbb = np.array(Image.fromarray(cell.orig[x_left:x_right, y_left:y_right]).resize((24, 24)))
        bbb = bbb/np.sum(bbb)
        bbbb = torch.from_numpy(bbb).type(torch.FloatTensor).to(device)
        print('prediction:', classifier((bbbb - mean)/std))
        return classifier((bbbb - mean)/std)

    return predictor


def find_threshold(refls, labels):
    pairs = list(zip(refls, labels))
    pairs.sort(key=lambda pair: pair[0])
    num_of_correct = np.sum(labels)
    best_threshold = 0
    best_correct = num_of_correct

    for refl, label in pairs:
        num_of_correct += -2*label + 1
        if num_of_correct > best_correct:
            best_correct = num_of_correct
            best_threshold = refl

    return best_threshold, best_correct

def threshold_accuracy(threshold, refls, labels):
    correct = 0
    for refl, label in zip(refls, labels):
        if (refl <= threshold and label == 0) or (refl > threshold and label == 1):
            correct += 1
    
    return correct/len(labels)

def validate(circles, alpha_lambda):
    generated_cells_directory = '../data/stemcells/simulated/other_initilization_'+ str(circles) +'x'+ str(circles) +'circles_20/optimized_vertex_lists_1.0_0.01_' + str(alpha_lambda)

    originals1 = [[]] + sorted(glob.glob('../data/stemcells/01/*.tif'))
    originals1 = reduce(lambda a, b: a + [imageio.imread(b)], originals1)
    originals1 = [originals1[20], originals1[30], originals1[32], originals1[38]]

    gts1 = [[]] + sorted(glob.glob('../data/stemcells/01_GT/SEG/man_seg???.tif'))
    gts1 = reduce(lambda a, b: a + [imageio.imread(b)], gts1)
    gts1 = [gts1[14], gts1[18], gts1[20], gts1[23]]

    originals2 = [[]] + sorted(glob.glob('../data/stemcells/02/*.tif'))
    originals2 = reduce(lambda a, b: a + [imageio.imread(b)], originals2)
    originals2 = [originals2[0], originals2[51], originals2[54], originals2[71]]

    gts2 = [[]] + sorted(glob.glob('../data/stemcells/02_GT/SEG/man_seg???.tif'))
    gts2 = reduce(lambda a, b: a + [imageio.imread(b)], gts2)
    gts2 = [gts2[0], gts2[11], gts2[12], gts2[15]]

    scenes2 = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=2)
    scenes2 = [scenes2[0], scenes2[51], scenes2[54], scenes2[71]]

    scenes1 = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=1)
    scenes1 = [scenes1[20], scenes1[30], scenes1[32], scenes1[38]]


    refl = np.load(generated_cells_directory + '/refl.npy')
    refl2 = np.load(generated_cells_directory + '/refl2.npy')
    labels = np.load(generated_cells_directory + '/labels.npy')
    labels2 = np.load(generated_cells_directory + '/labels2.npy')
    llabels = np.concatenate([labels, labels])
    cell_imgs = np.load(generated_cells_directory + '/cellimages.npy')
    train_len = len(cell_imgs)
    cell_imgs2 = np.load(generated_cells_directory + '/cellimages2.npy')
    cell_imgs = np.concatenate([cell_imgs, np.swapaxes(cell_imgs, 1, 2)[:, ::-1]])
    print(train_len, len(cell_imgs))

    threshold, correct = find_threshold(refl, labels)
    threshold2, correct2 = find_threshold(refl2, labels2)
    print(len(cell_imgs))
    print(len(labels), len(labels2))
    print(threshold)
    print(threshold_accuracy(threshold, refl, labels))
    print(threshold_accuracy(threshold, refl2, labels2))
    print(threshold2)
    print(threshold_accuracy(threshold2, refl, labels))
    print(threshold_accuracy(threshold2, refl2, labels2))
    print(np.sum(labels2))
    print(np.sum(labels2)/len(labels2))
    print(np.sum(labels)/len(labels))
    print(np.sum(np.isnan(cell_imgs)), np.max(cell_imgs))
    cell_imgs = (1-np.isnan(cell_imgs)) * cell_imgs
    #print(labels[100:])
    if plotit:
        plt.plot(np.extract(labels, refl), np.ones(int(np.sum(labels))), 'o')
        plt.plot(np.extract(labels == False, refl), np.zeros(int(len(labels) - np.sum(labels))), 'ro')
        plt.show()
        plt.plot(np.extract(labels2, refl2), np.ones(int(np.sum(labels2))), 'o')
        plt.plot(np.extract(labels2 == False, refl2), np.zeros(int(len(labels2) - np.sum(labels2))), 'ro')
        plt.show()

    # NN
    mean = np.mean(cell_imgs)
    std = np.std(cell_imgs)
    print(mean, std)
    cell_imgs -= mean
    cell_imgs /= std
    net = train.simple_nn(cell_imgs, llabels, train_len, early_stopping_patience=100)

    if plotit:
        for i in range(-1, -10, -1):
            print(llabels[i])
            print(net(torch.from_numpy(cell_imgs[i]).type(torch.FloatTensor).to(device).reshape(1, 1, 24, 24)))
            plt.imshow(cell_imgs[i])
            plt.show()


    def nn_classifier(crop):
        output = net(crop.reshape(1, 1, 24, 24))
        print(output)
        pred = output[0, 0] < 0.5
        print(pred)
        return pred.item()

    stats1 = evaluate(
        scenes1,
        gts1,
        originals1,
        thresholding(threshold),
        crop_predictor(nn_classifier, mean, std)
    )

    stats2 = evaluate(
        scenes2,
        gts2,
        originals2,
        thresholding(threshold),
        crop_predictor(nn_classifier, mean, std)
    )

    with open(generated_cells_directory + '/test_info2.csv', 'w') as info:
        wr = csv.writer(info, delimiter=';')
        wr.writerow([str(a) for a in stats1])
        wr.writerow([str(a) for a in stats2])
    print(stats1)
    print(stats2)

plotit = False
for circles in range(7, 14):
    for alpha_lambda in [0, 0.5, 1, 5, 10, 50]:
        validate(circles, alpha_lambda)