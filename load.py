import torch
import glob
import random
import imageio
from functools import reduce
from cell import Cell, pose_matrices, positions, render_simulation
import matplotlib.pyplot as plt
from setup import device
import skimage

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

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell(d[1], d[0], 0)], [[]] + list(zip(b[0], b[1])))], cells)

def simulated_cell_vertices(path):
    vertices = [[]]

    vertices += sorted(glob.glob(path))
    vertices = reduce(lambda a, b: a + [tensor_to_list(torch.load(b, map_location=device))], vertices)

    return reduce(lambda a, b: a + [reduce(lambda c, d: c + [Cell.from_vertices(d)], [[]] + b)], [[]] + vertices)

class RotatingSelectiveLoader:

    # batchsize will effectively be rounded up to the next number divisible by four
    def __init__(self, pathtocelldata, pathtoimages, batch_size, width, height):
        self.width = width
        self.height = height
        self.center = torch.FloatTensor([width/2, height/2]).to(device)
        self.simulated_ellipses = simulated_ellipses(pathtocelldata)
        self.imgpaths = sorted(glob.glob(pathtoimages))
        print(len(self.imgpaths), len(self.simulated_ellipses))
        self.batch_size = (batch_size+3)//4

    def __iter__(self):
        self.experiment = 0
        self.cell = 0
        return self

    def __len__(self):
        return 4*reduce(lambda a, b: a + len(b), [0] + self.simulated_ellipses)

    def __next__(self):
        rotation90 = torch.FloatTensor([[0, -1], [1, 0]]).to(device)
        rotation270 = torch.FloatTensor([[0, 1], [-1, 0]]).to(device)

        raw_image = imageio.imread(self.imgpaths[self.experiment])
        img = torch.from_numpy(skimage.transform.rescale(raw_image,
                (self.width/raw_image.shape[0], self.height/raw_image.shape[1]))).float().to(device)
        input_batch = []
        target_batch = []
        i = 0
        while i < self.batch_size:
            if (self.experiment == len(self.simulated_ellipses)):
                break
            if (self.cell == len(self.simulated_ellipses[self.experiment])):
                self.experiment += 1
                raw_image = imageio.imread(self.imgpaths[self.experiment])
                img = torch.from_numpy(skimage.transform.rescale(raw_image,
                        (self.width/raw_image.shape[0], self.height/raw_image.shape[1]))).float().to(device)
                self.cell = 0
            else:
                input_pos = self.simulated_ellipses[self.experiment][self.cell].position
                input_pose = self.simulated_ellipses[self.experiment][self.cell].pose_matrix
                input_batch.append(torch.cat([input_pos, torch.flatten(input_pose)]))
                #plt.figure(1)
                #plt.imshow(render_simulation([self.simulated_ellipses[self.experiment][self.cell]]).detach())
                #plt.figure(2)
                #plt.imshow(img)
                #plt.show()
                target = (0 < render_simulation([self.simulated_ellipses[self.experiment][self.cell]])) * img
                target_batch.append(target)
                for j in range(3):
                    input_pos = torch.matmul(rotation90, input_pos - self.center) + self.center
                    input_pose = torch.matmul(rotation270, input_pose)
                    input_batch.append(torch.cat([input_pos, torch.flatten(input_pose)]))
                    target = target.flip(-1).t()
                    target_batch.append(target)
                self.cell += 1
                i += 1
        if(not input_batch):
            raise StopIteration
        return torch.stack(input_batch), torch.stack(target_batch)
