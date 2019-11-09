import torch
import glob
from functools import reduce
from cell import Cell, pose_matrices, positions
import matplotlib.pyplot as plt
from setup import device

def tensor_to_list(tensor):
    l = []
    for i in range(len(tensor)):
        l.append(tensor[i])
    return l

def simulated_ellipses():
    positions = [[]]
    pose_matrices = [[]]

    positions += sorted(glob.glob('../data/stemcells/simulated/???pos.pt'))
    pose_matrices += sorted(glob.glob('../data/stemcells/simulated/???pose.pt'))

    positions = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], positions)
    pose_matrices = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], pose_matrices)

    cells = [[]] + list(zip(positions, pose_matrices))

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell(d[1], d[0])], [[]] + list(zip(b[0], b[1])))], cells)
    