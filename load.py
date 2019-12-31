import torch
import glob
import random
import imageio
from functools import reduce
from cell import Cell, pose_matrices, positions
import matplotlib.pyplot as plt
from setup import device

def tensor_to_list(tensor):
    l = []
    for i in range(len(tensor)):
        l.append(tensor[i])
    return l

def simulated_ellipses(path=''):
    positions = [[]]
    pose_matrices = [[]]

    positions += sorted(glob.glob('../data/stemcells/simulated/'+path+'/???pos.pt'))
    pose_matrices += sorted(glob.glob('../data/stemcells/simulated/'+path+'/???pose.pt'))

    positions = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], positions)
    pose_matrices = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], pose_matrices)

    cells = [[]] + list(zip(positions, pose_matrices))

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell(d[1], d[0])], [[]] + list(zip(b[0], b[1])))], cells)

def simulated_cell_vertices(path):
    vertices = [[]]

    vertices += sorted(glob.glob(path))
    vertices = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], vertices)

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell.from_vertices(d)], [[]] + b)], [[]] + vertices)

class TransformingLoader:

    def __init__(self, pathtocelldata, pathtoimages, batch_size, width, height):
        self.width = width
        self.height = height
        self.celldatapaths = sorted(glob.glob(pathtocelldata))
        self.imgpaths = sorted(glob.glob(pathtoimages))
        assert(len(self.imgpaths) == len(self.celldatapaths))
        self.batch_size = batch_size
        self.transform = lambda img, target: (img, target)

    def add_transform(self, transform, option_count):
        prev_transform = self.transform
        def rand_transform(img, target):
            r = int(random.uniform(1, option_count+1))

            return transform(*prev_transform(img, target), r)
        self.transform = rand_transform

    def next_batch(self):
        inputbatch = []
        targetbatch = []
        for i in range(self.batch_size):
            r = random.randrange(len(self.imgpaths))
            vertices = torch.load(self.celldatapaths[r], map_location=device)
            img = torch.from_numpy(skimage.transform.rescale(imageio.imread(self.imgpaths[r]),
                (self.width/raw_image.shape[0], self.height/raw_image.shape[1]))).float().to(device)
            vertices, img = self.transform(vertices, img)
            # TODO create masked img and apprpriate target from vertices
            inputbatch.append(img)
            targetbatch.append(vertices)
        return torch.stack(inputbatch), torch.stack(targetbatch)
