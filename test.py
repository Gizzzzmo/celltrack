import torch
import numpy as np
import sys
import imageio
import glob
import torchvision
import skimage
from cell import Cell, positions, pose_matrices, render_simulation, redner_simulation
from matplotlib import pyplot as plt
from setup import device, xy, width, height

print(device)

def createCells():
    cells = []
    for i in range(width//5, 4*width//5, width//5):
        for j in range(height//5, 4*height//5, width//5):
            print(i, j)
            pos = torch.FloatTensor([i, j]).to(device)
            M = torch.FloatTensor([[3.6e-02, 1.3e-05], [1.3e-05, 3.6e-02]]).to(device)
            M.requires_grad=True
            pos.requires_grad=True
            cells.append(Cell(M, pos))
    return cells

def loss_fn(cells, target, stage=1):
    simulated = render_simulation(cells, stage)
    return (target-simulated), simulated

def plot(diff, simulated):
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(diff.cpu().abs().detach().numpy())
    plt.subplot(212)
    plt.imshow(simulated.cpu().detach().numpy())
    
    plt.show()

def optimize_position(iterations, target, stage=2):
    print('optimizing position')
    for i in range(iterations):
        optimizer1.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        if(i%100 == 99):
            print('plotting...')
            plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer1.step()

def optimize_pose(iterations, target, stage=2):
    print('optimizing pose...')
    for i in range(iterations):
        optimizer2.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        if(i%100 == 99):
            print('plotting...')
            plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer2.step()

def optimize_vertices(iterations, target, optimizer):
    print('optimizing vertices for rendering...')
    for i in range(iterations):
        optimizer.zero_grad()
        img = redner_simulation(cells)
        loss = (target-img).pow(2).sum()
        loss.backward(retain_graph=True)
        optimizer.step()


def split(threshold):
    print('splitting...')
    for cell in cells:
        u, s, v = torch.svd(cell.pose_matrix)
        ecc = 1-(s[1]/s[0])**2
        if(ecc > threshold**2 and cell.visible):
            cell.delete()
            M1 = torch.FloatTensor([[1/16, 0], [0, 1/16]]).to(device)
            M2 = torch.FloatTensor([[1/16, 0], [0, 1/16]]).to(device)
            pos = cell.position
            pos.requires_grad = False
            offset = torch.FloatTensor([-u[0, 1], u[0, 0]]).to(device)/s[0]
            pos1 = (pos+offset).detach()
            pos2 = (pos-offset).detach()
            M1.requires_grad = True
            M2.requires_grad = True
            pos1.requires_grad = True
            pos2.requires_grad = True
            cells.append(Cell(M1, pos1))
            cells.append(Cell(M2, pos2))

def delete_superfluous(threshold, target):
    print('deleting...')
    for cell in cells:
        diff, simulated = loss_fn([cell], target, 2)
        loss = diff.pow(2).sum()
        norm = target.pow(2).sum()
        if(loss > norm*threshold or simulated.sum() < 1e-1):
            cell.delete()

def create_polygons(cells, threshold):
    print('creating vertex lists for polygons...')
    for cell in cells:
        if cell.visible:
            cell.create_shape(threshold)
        del cell.pose_matrix
    torch.cuda.empty_cache()

i = 0
print("go")
for path in sorted(glob.glob('../data/stemcells/closed01/*.png')):
    print(path)
    if(i == 0):
        i += 1
        continue
    raw_image = imageio.imread(path)
    
    target = 255*torch.from_numpy(skimage.transform.rescale(raw_image,
        (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)
    cells = createCells()
    
    diff, simulated = loss_fn(cells, target)
    optimizer1 = torch.optim.Adam(positions(cells), lr=5)
    optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=2e-5)
    optimize_position(200, target, 1)
    optimize_pose(200, target)
    delete_superfluous(1.05, target)
    if(positions(cells)):
        split(0.65)
        optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=2e-5)
        optimizer1 = torch.optim.Adam(positions(cells), lr=5)
        optimize_position(100, target)
        optimize_pose(200, target)
        split(0.65)
        optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=2e-5)
        optimizer1 = torch.optim.Adam(positions(cells), lr=5)
        optimize_position(100, target)
        optimize_pose(200, target)
        split(0.65)
        optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=2e-5)
        optimizer1 = torch.optim.Adam(positions(cells), lr=5)
        optimize_position(100, target)
        optimize_pose(200, target)
        delete_superfluous(1, target)

        pos_array = torch.stack(positions(cells))
        pose_array = torch.stack(pose_matrices(cells))
        create_polygons(cells, 50)
        imgg = redner_simulation(cells)

        plot(imgg, imgg)

        #torch.save(pos_array, 'data/stemcells/simulated/'+f'{i:03}'+'pos.pt')
        #torch.save(pose_array, 'data/stemcells/simulated/'+f'{i:03}'+'pose.pt')

        #simulated = render_simulation(cells, stage=2)
        #torch.save(simulated, 'data/stemcells/simulated/'+f'{i:03}'+'simulatedimg.pt')

        del simulated
        del pose_array
        del pos_array

    del cells
    del target
    torch.cuda.empty_cache()
    i += 1
