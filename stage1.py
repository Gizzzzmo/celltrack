import torch
import numpy as np
import sys
import os
import imageio
import glob
import torchvision
import skimage
from cell import Cell, positions, pose_matrices, render_simulation
from matplotlib import pyplot as plt, animation
from setup import device, xy, width, height

print(device)

factor = 1
splitting_eccentricity = 0.71
pos_lr = 4
pose_lr = 5e-5
simulation_sequence = []
loss_sequence = []
createanimation = False
plotit = False

def createCells():
    cells = []
    for i in range(width//7, 6*width//7, width//7):
        for j in range(height//7, 6*height//7, width//7):
            pos = torch.FloatTensor([i, j]).to(device)
            M = (2**factor)*torch.FloatTensor([[3.6e-02, 1.3e-05], [1.3e-05, 3.6e-02]]).to(device)
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
        simulation_sequence.append(simulated.detach().cpu().numpy())
        loss_sequence.append(diff.abs().detach().cpu().numpy())
        if(i%100 == 99) and plotit:
            print('plotting...')
            plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer1.step()

def optimize_pose(iterations, target, stage=2):
    print('optimizing pose...')
    for i in range(iterations):
        optimizer2.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        if(createanimation):
            simulation_sequence.append(simulated.detach().cpu().numpy())
            loss_sequence.append(diff.abs().detach().cpu().numpy())
        loss = diff.pow(2).sum()
        if(i%100 == 99) and plotit:
            print('plotting...')
            plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer2.step()

def split(threshold):
    print('splitting...')
    for cell in cells:
        u, s, v = torch.svd(cell.pose_matrix)
        ecc = 1-(s[1]/s[0])**2
        if(ecc > threshold**2 and cell.visible):
            cell.delete()
            M1 = (2**factor)*torch.FloatTensor([[1/16, 0], [0, 1/16]]).to(device)
            M2 = (2**factor)*torch.FloatTensor([[1/16, 0], [0, 1/16]]).to(device)
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

i = 0
print("go")
for path in sorted(glob.glob('../data/stemcells/closed01/*.png')):
    print(path)
    if(i < 56):
        i += 1
        continue
    raw_image = imageio.imread(path)
    
    target = 255*torch.from_numpy(skimage.transform.rescale(raw_image,
        (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)
    cells = createCells()
    
    diff, simulated = loss_fn(cells, target)
    #plot(diff, simulated)
    optimizer1 = torch.optim.Adam(positions(cells), lr=pos_lr)
    optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=pose_lr)
    optimize_position(200, target, 1)
    optimize_pose(200, target)
    delete_superfluous(1.05, target)
    if(positions(cells)):
        split(splitting_eccentricity)
        optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=pose_lr)
        optimizer1 = torch.optim.Adam(positions(cells), lr=pos_lr)
        optimize_position(100, target)
        optimize_pose(200, target)
        split(splitting_eccentricity)
        optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=pose_lr)
        optimizer1 = torch.optim.Adam(positions(cells), lr=pos_lr)
        optimize_position(100, target)
        optimize_pose(500, target)
        split(splitting_eccentricity)
        optimizer2 = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=pose_lr)
        optimizer1 = torch.optim.Adam(positions(cells), lr=pos_lr)
        optimize_position(100, target)
        optimize_pose(500, target)
        delete_superfluous(1, target)

        target_dir = '../data/stemcells/simulated/'+str(width)+'x'+str(height)+'_'+str(splitting_eccentricity)+'_'+str(pos_lr)+'_'+str(pose_lr)+'/'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if createanimation:
            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255)
            def animate(i):
                factor = 255.0/np.max(simulation_sequence[i])
                im.set_array(factor* simulation_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(simulation_sequence), interval=1, blit=True)
            anim.save(target_dir + 'simulated_blobs_' + str(i) +'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.show()

            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255)
            def animate(i):
                factor = 255.0/np.max(loss_sequence[i])
                im.set_array(factor* loss_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(loss_sequence), interval=1, blit=True)
            anim.save(target_dir + 'simulated_blobs_loss_' + str(i) +'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.show()

        pos_array = torch.stack(positions(cells))
        pose_array = torch.stack(pose_matrices(cells))
        torch.save(pos_array, target_dir+f'{i:03}'+'pos.pt')
        torch.save(pose_array, target_dir+f'{i:03}'+'pose.pt')

        if(createanimation):
            simulation_sequence = []
            loss_sequence = []
        del pose_array
        del pos_array

    del cells
    del target
    torch.cuda.empty_cache()
    i += 1
