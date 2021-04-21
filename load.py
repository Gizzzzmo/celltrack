import torch
import glob
from functools import reduce
from cell import Cell
from setup import device

def tensor_to_list(tensor):
    """ turns a tensor into a list of all the non top-level tensors"""
    l = []
    for i in range(len(tensor)):
        l.append(tensor[i])
    return l

def simulated_ellipses(path=''):
    """ loads simulated ellipses from the given path and returns them in a list of cells """
    positions = [[]]
    pose_matrices = [[]]

    positions += sorted(glob.glob('../data/stemcells/simulated/'+path+'/???pos.pt'))
    pose_matrices += sorted(glob.glob('../data/stemcells/simulated/'+path+'/???pose.pt'))

    positions = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], positions)
    pose_matrices = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], pose_matrices)

    cells = [[]] + list(zip(positions, pose_matrices))

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell(d[1], d[0], 0)], [[]] + list(zip(b[0], b[1])))], cells)

def simulated_cell_vertices(path, series=1):
    """ loads vertex lists from the given path and the time series and returns them in a list of cells"""
    series = '' if series == 1 else str(series)
    vertices = [[]]

    vertices += sorted(glob.glob(path + '/???vert' + series + '.pt'))
    vertices = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], vertices)

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell.from_vertices(d)], [[]] + b)], [[]] + vertices)

def simulated_cell_vertices_plus_reflectances(path, series=1):
    """ loads vertex lists and reflectances from the given path and the time series and returns them in a list of cells"""
    scenes = simulated_cell_vertices(path, series)
    series = '' if series == 1 else str(series)

    reflectances = [[]]
    reflectances += sorted(glob.glob(path + '/???refl' + series + '.pt'))
    reflectances = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], reflectances)
    for cells, refl in zip(scenes, reflectances):
        for c, r in zip(cells, refl):
            c.diffuse_reflectance = r
    return scenes
