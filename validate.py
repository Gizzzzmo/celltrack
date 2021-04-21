from setup import alpha, device, xy
import pyredner
import torch
import glob
import imageio
import numpy as np
from PIL import Image
from functools import reduce
from sortedcontainers import SortedList
from load import simulated_cell_vertices_plus_reflectances
from scene import create_scene, create_scene_environment
import train
import csv
import viewsim
import matplotlib.pyplot as plt


def update_assigned_cell(cell, mask_to_cell_map, classifier_index):
    """ update the simulated cell assigned to a reference segment """
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
    """
        evaluates a number of *classifiers*, on a list of *scenes* and the corresponding *original* images,
        using a list of *target_masks*
    """

    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    classifiers = list(classifiers)
    classifiers.append(lambda _: True)

    avg_jaccard = [0] * len(classifiers)
    num_identified_cells = [0] * len(classifiers)
    false_positives = [0] * len(classifiers)
    missed_cells = [0] * len(classifiers)
    num_cells = 0

    test_reflectance = torch.cuda.FloatTensor([100, 100, 100])

    for cells, target_mask, orig in zip(scenes, target_masks, originals):
        print('next')
        mask_to_cell_map = [{} for _ in avg_jaccard]
        caught_cells = [0] * len(classifiers)
        pruned_lists = [[] for _ in classifiers]
        for cell in cells:
            
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
                plt.show()
            cell.jaccard = [None] * len(classifiers)

            cell.pred = [None] * len(classifiers)
            for i, predict in enumerate(classifiers):
                if predict(cell):
                    pruned_lists[i].append(cell)
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
            viewsim.seg_mask1(pruned_lists[0], orig, 'pruned_mask_0.png')
            viewsim.seg_mask1(pruned_lists[1], orig, 'pruned_mask_1.png')
            viewsim.seg_mask1(pruned_lists[2], orig, 'pruned_mask_2.png')
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
                        jaccard, _ = cell.jaccard[i][-1]
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
    """ turns a diffuse reflectance based classifier into a decision function """
    return lambda cell: classifier.predict(cell.diffuse_reflectance[0].detach().cpu().numpy().reshape(1, 1))

def thresholding(threshold):
    """ turns a diffuse reflectance based threshold into a decision function """
    return lambda cell: cell.diffuse_reflectance[0].item() > threshold

def crop_predictor(classifier, mean, std):
    """ turns an image cutout based classier into cell based classifier, also makes sure that data is properly normalized """
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
        bbbb = torch.from_numpy(bbb).type(torch.FloatTensor).to(device)
        print('prediction:', classifier((bbbb - mean)/std))
        return classifier((bbbb - mean)/std)

    return predictor


def find_threshold(refls, labels):
    """ finds the threshold for a list of reflectances and their labels that produces the best accuracy """
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
    """computes the accuracy of a given threshold"""
    correct = 0
    for refl, label in zip(refls, labels):
        if (refl <= threshold and label == 0) or (refl > threshold and label == 1):
            correct += 1
    
    return correct/len(labels)

def validate(circles, alpha_lambda):
    """
    Loads saved scenes that were the result of a *circles*x*circles* starting layout using a regularization weight of *alpha_lambda*
    as well as the resulting cutouts from the original images.
    Part of the data is used to train a neural network and find a threshold based on the simulated cell's diffuse reflectances.
    The resulting classifiers (who judge whether a simulated cell actually covers a real one) are then evaluated on the rest of the data.
    Metrics are printed to the console and saved in a csv file
    """
    generated_cells_directory = '../data/stemcells/simulated/other_initilization_'+ str(circles) +'x'+ str(circles) +'circles_20/optimized_vertex_lists_1.0_0.01_' + str(alpha_lambda)

    # load the target images from time series 1
    originals1 = [[]] + sorted(glob.glob('../data/stemcells/01/*.tif'))
    originals1 = reduce(lambda a, b: a + [imageio.imread(b)], originals1)
    # get the four fully segmented ones
    originals1 = [originals1[20], originals1[30], originals1[32], originals1[38]]

    # load the ground truth segmentation masks for these four images
    gts1 = [[]] + sorted(glob.glob('../data/stemcells/01_GT/SEG/man_seg???.tif'))
    gts1 = reduce(lambda a, b: a + [imageio.imread(b)], gts1)
    gts1 = [gts1[14], gts1[18], gts1[20], gts1[23]]

    # repeat the above steps for time series 2
    originals2 = [[]] + sorted(glob.glob('../data/stemcells/02/*.tif'))
    originals2 = reduce(lambda a, b: a + [imageio.imread(b)], originals2)
    originals2 = [originals2[0], originals2[51], originals2[54], originals2[71]]

    gts2 = [[]] + sorted(glob.glob('../data/stemcells/02_GT/SEG/man_seg???.tif'))
    gts2 = reduce(lambda a, b: a + [imageio.imread(b)], gts2)
    gts2 = [gts2[0], gts2[11], gts2[12], gts2[15]]

    # load the corresponding scenes (coming from stage1) for all the images
    scenes2 = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=2)
    scenes2 = [scenes2[0], scenes2[51], scenes2[54], scenes2[71]]

    scenes1 = simulated_cell_vertices_plus_reflectances(generated_cells_directory, series=1)
    scenes1 = [scenes1[20], scenes1[30], scenes1[32], scenes1[38]]

    # load the training data for the thresholding and the neural network (coming from stage2)
    refl = np.load(generated_cells_directory + '/refl.npy')
    refl2 = np.load(generated_cells_directory + '/refl2.npy')
    labels = np.load(generated_cells_directory + '/labels.npy')
    labels2 = np.load(generated_cells_directory + '/labels2.npy')
    llabels = np.concatenate([labels, labels])
    cell_imgs = np.load(generated_cells_directory + '/cellimages.npy')
    train_len = len(cell_imgs)
    cell_imgs = np.concatenate([cell_imgs, np.swapaxes(cell_imgs, 1, 2)[:, ::-1]]).astype(float)
    print(train_len, len(cell_imgs))

    # find the optimally separating threshold once for the data from the first and once for the data from the second time series
    threshold, _ = find_threshold(refl, labels)
    threshold2, _ = find_threshold(refl2, labels2)
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
    cell_imgs = (1-np.isnan(cell_imgs)) * cell_imgs
    
    if plotit:
        plt.plot(np.extract(labels, refl), np.ones(int(np.sum(labels))), 'o')
        plt.plot(np.extract(labels == False, refl), np.zeros(int(len(labels) - np.sum(labels))), 'ro')
        plt.show()
        plt.plot(np.extract(labels2, refl2), np.ones(int(np.sum(labels2))), 'o')
        plt.plot(np.extract(labels2 == False, refl2), np.zeros(int(len(labels2) - np.sum(labels2))), 'ro')
        plt.show()

    # train the CNN on the cell cutouts from stage2
    # first estimating mean and standard deviation
    mean = np.mean(cell_imgs)
    std = np.std(cell_imgs)
    print(mean, std)
    # to then normalize
    cell_imgs -= mean
    cell_imgs /= std
    # and finally train the network
    net = train.simple_nn(cell_imgs, llabels, train_len, early_stopping_patience=100)

    # plot some of the cell cutouts and the nn predicted label
    if plotit:
        for i in range(-1, -10, -1):
            print(llabels[i])
            print(net(torch.from_numpy(cell_imgs[i]).type(torch.FloatTensor).to(device).reshape(1, 1, 24, 24)))
            plt.imshow(cell_imgs[i])
            plt.show()

    # define a function that takes in a cropped image and predicts, with the help of the network, the label
    def nn_classifier(crop):
        output = net(crop.reshape(1, 1, 24, 24))
        print(output)
        pred = output[0, 0] < 0.5
        print(pred)
        return pred.item()

    # evaluate the performance of the network and the threshold on the time series they were trained on
    stats1 = evaluate(
        scenes1,
        gts1,
        originals1,
        thresholding(threshold),
        crop_predictor(nn_classifier, mean, std)
    )

    # evaluate the performance of the network and the threshold on the time series they weren't trained on
    stats2 = evaluate(
        scenes2,
        gts2,
        originals2,
        thresholding(threshold),
        crop_predictor(nn_classifier, mean, std)
    )

    # save and display all results
    with open(generated_cells_directory + '/test_info2.csv', 'w') as info:
        wr = csv.writer(info, delimiter=';')
        wr.writerow([str(a) for a in stats1])
        wr.writerow([str(a) for a in stats2])
    print(stats1)
    print(stats2)

# toggle to show additional plots during the validation process
plotit = True
# validate combinations of starting layouts and regularization weights
for circles in range(11, 12):
    for alpha_lambda in [1]:
        validate(circles, alpha_lambda)