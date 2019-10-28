import torch
import numpy as np
import imageio
import glob
import torchvision
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

width = 1024
height = 1024

xx = torch.arange(0, width, 1, device=device, dtype=torch.float32)
yy = torch.arange(0, height, 1, device=device, dtype=torch.float32)
xxx = xx.expand((height, -1))
yyy = yy.expand((width, -1))
yyy = yyy.transpose(0, 1)

xy = torch.stack([xxx, yyy], dim=1)


class Cell:

    def __init__(self, pose_matrix, position):
        self.visible = True
        self.pose_matrix = pose_matrix
        self.position = position

        self.b = position.expand(1, -1).transpose(0, 1)
    
    def render(self, x, stage=1):
        diff = x-self.b
        transformed = torch.matmul(self.pose_matrix, diff)
        transformed_dist = torch.sum(transformed.pow(2), dim=(-2,)).pow(stage)

        return 255*torch.exp(-transformed_dist)
    
    def delete(self):
        self.visible = False

def createCells():
    cells = []
    for i in range(width//5+1, 4*width//5, width//5):
        for j in range(height//5+1, 4*height//5, width//5):
            pos = torch.cuda.FloatTensor([i, j])
            M = torch.cuda.FloatTensor([[1.5385e-02, 6.2500e-06], [6.2500e-06, 1.5385e-02]])
            M.requires_grad=True
            pos.requires_grad=True
            cells.append(Cell(M, pos))
    return cells

def render_simulation(cells, target, stage=1):
    simulated = torch.zeros_like(target)
    for cell in cells:
        if(cell.visible):
            simulated += cell.render(xy, stage)

    return simulated

def loss_fn(cells, target, stage=1):
    simulated = render_simulation(cells, target, stage)
    return (target-simulated), simulated

def plot(diff, simulated):
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(diff.cpu().abs().detach().numpy())
    plt.subplot(212)
    plt.imshow(simulated.cpu().detach().numpy())
    
    plt.show()

def positions(cells):
    return [cell.position for cell in cells if cell.visible]

def pose_matrices(cells):
    return [cell.pose_matrix for cell in cells if cell.visible]

def optimize_position(iterations, target, stage=2):
    print('optimizing position')
    for i in range(iterations):
        optimizer1.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        if(i%100 == 99):
            print(cells[0].pose_matrix)
            #plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer1.step()

def optimize_pose(iterations, target, stage=2):
    print('optimizing pose...')
    for i in range(iterations):
        optimizer2.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        if(i%100 == 99):
            print(cells[0].pose_matrix)
            #plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer2.step()

def split(threshold):
    print('splitting...')
    for cell in cells:
        u, s, v = torch.svd(cell.pose_matrix)
        ecc = 1-(s[1]/s[0])**2
        if(ecc > threshold**2 and cell.visible):
            cell.delete()
            M1 = torch.cuda.FloatTensor([[1/32, 0], [0, 1/32]])
            M2 = torch.cuda.FloatTensor([[1/32, 0], [0, 1/32]])
            pos = cell.position
            pos.requires_grad = False
            offset = torch.cuda.FloatTensor([-u[0, 1], u[0, 0]])/s[0]
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

i = 0
for path in sorted(glob.glob('data/stemcells/closed01/*.png')):
    print(path)
    target = torch.from_numpy(imageio.imread(path)).float().to(device)

    cells = createCells()
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
        delete_superfluous(1, target)
        pos_array = torch.stack(positions(cells))
        pose_array = torch.stack(pose_matrices(cells))

        torch.save(pos_array, 'data/stemcells/simulated/'+str(i)+'pos.pt')
        torch.save(pose_array, 'data/stemcells/simulated/'+str(i)+'pose.pt')

        simulated = render_simulation(cells, target, stage=2)
        torch.save(simulated, 'data/stemcells/simulated/'+str(i)+'simulatedimg.pt')

        del simulated
        del pose_array
        del pos_array

    del cells
    del target
    torch.cuda.empty_cache()
    i += 1
