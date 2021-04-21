import torch
import numpy as np
import os
import imageio
import glob
import skimage
from cell import Cell, positions, pose_matrices, brightnesses, render_simulation
from matplotlib import pyplot as plt, animation
from setup import device, width, height, density_diagram

print(device)

factor = 2.8
splitting_eccentricity = 0.71
pos_lr = 4
pose_lr = 5e-5
brightness_lr = 5e-2
simulation_sequence = []
loss_sequence = []
createanimation = False
plotit = False
preprocessed = False

def createCells():
    cells = []
    for i in range(width//9, 8*width//9 + 1, width//9):
        for j in range(height//9, 8*height//9 + 1, width//9):
            pos = torch.FloatTensor([i, j]).to(device)
            M = factor*torch.FloatTensor([[3.6e-02, 1.3e-05], [1.3e-05, 3.6e-02]]).to(device)
            M.requires_grad=True
            pos.requires_grad=True
            cells.append(Cell(M, pos, 0))
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

def optimize(iterations, target, optimizer, stage=2):
    for i in range(iterations):
        optimizer.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        if(createanimation):
            detached = (simulated.detach().cpu().numpy(), diff.abs().detach().cpu().numpy())
            for c in cells:
                x_pos = int(c.position.detach().cpu().numpy()[1])
                y_pos = int(c.position.detach().cpu().numpy()[0])
                if (0 <= x_pos < width) and (0 <= y_pos < height) and c.visible:
                    detached[0][x_pos, y_pos] = 200
                    detached[1][x_pos, y_pos] = 200
            simulation_sequence.append(detached[0])
            loss_sequence.append(detached[1])
        if(i%100 == 99) and plotit:
            print('plotting...')
            plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer.step()

def optimize_position(iterations, target, cells, stage=2):
    print('optimizing position')
    optimizer = torch.optim.Adam(positions(cells), lr=pos_lr)
    optimize(iterations, target, optimizer, stage)

def optimize_brightness(iterations, target, cells, stage=2):
    print('optimizing brightness')
    optimizer = torch.optim.Adam(brightnesses(cells), lr=brightness_lr)
    optimize(iterations, target, optimizer, stage)

def optimize_pose(iterations, target, cells, stage=2):
    print('optimizing pose...')
    optimizer = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=pose_lr)
    optimize(iterations, target, optimizer, stage)

def optimize_position_and_brightness(iterations, target, cells, stage=2):
    optimizer1 = torch.optim.Adam(positions(cells), lr=pos_lr)
    optimizer2 = torch.optim.Adam(brightnesses(cells), lr=brightness_lr)
    for i in range(iterations):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        if(createanimation):
            detached = (simulated.detach().cpu().numpy(), diff.abs().detach().cpu().numpy())
            for c in cells:
                x_pos = int(c.position.detach().cpu().numpy()[1])
                y_pos = int(c.position.detach().cpu().numpy()[0])
                if (0 <= x_pos < width) and (0 <= y_pos < height) and c.visible:
                    detached[0][x_pos, y_pos] = 200
                    detached[1][x_pos, y_pos] = 200
            simulation_sequence.append(detached[0])
            loss_sequence.append(detached[1])
        if(i%100 == 99) and plotit:
            print('plotting...')
            plot(diff, simulated)
        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()

def split(threshold):
    print('splitting...')
    for cell in cells:
        u, s, v = torch.svd(cell.pose_matrix)
        ecc = 1-(s[1]/s[0])**2
        if(ecc > threshold**2 and cell.visible):
            cell.delete()
            M1 = factor*torch.FloatTensor([[1/16, 0], [0, 1/16]]).to(device)
            M2 = factor*torch.FloatTensor([[1/16, 0], [0, 1/16]]).to(device)
            pos = cell.position
            pos.requires_grad = False
            offset = torch.FloatTensor([-u[0, 1], u[0, 0]]).to(device)/s[0]
            pos1 = (pos+offset).detach()
            pos2 = (pos-offset).detach()
            M1.requires_grad = True
            M2.requires_grad = True
            pos1.requires_grad = True
            pos2.requires_grad = True
            cells.append(Cell(M1, pos1, cell.brightness[0]))
            cells.append(Cell(M2, pos2, cell.brightness[0]))

def delete_superfluous(threshold, target):
    print('deleting...')
    for cell in cells:
        if cell.visible:
            diff, simulated = loss_fn([cell], target, 2)
            loss = diff.pow(2).sum()
            norm = target.pow(2).sum()
            if(loss > norm*threshold or simulated.sum() < 1e-1):
                print('deleted', cell.position)
                cell.delete()

def wiggled_gradients(target, stage=2):
    gradients = []
    for cell in cells:
        if cell.visible:
            print(cell.position)
            cell.gradient_sum = 0
            for offset in [-1, 1]:
                cell.position.data[0] += offset

                diff = loss_fn([cell], target, stage)[0]
                loss = diff.pow(2).sum()
                cell.position.grad = None
                loss.backward(retain_graph=True)
                cell.gradient_sum += cell.position.grad.abs().sum()

                cell.position.data[0] -= offset
                cell.position.data[1] += offset

                diff = loss_fn([cell], target, stage)[0]
                loss = diff.pow(2).sum()
                cell.position.grad = None
                loss.backward(retain_graph=True)
                cell.gradient_sum += cell.position.grad.abs().sum()

                cell.position.data[1] -= offset
            
            gradients.append(cell.gradient_sum.item())
    
    return gradients

i = 0
target_dir = '../data/stemcells/closed01/*.png' if preprocessed else '../data/stemcells/01/*.tif'
print("go")
for path in sorted(glob.glob(target_dir)):
    print(path)
    if(i < 1):
        i += 1
        continue
    raw_image = imageio.imread(path)
    
    target = 255*torch.from_numpy(skimage.transform.rescale(raw_image,
        (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)
    cells = createCells()
    
    wiggled = wiggled_gradients(target)
    x, y = density_diagram(wiggled)
    plt.plot(x, y)
    plt.plot(np.array(wiggled), np.zeros((len(wiggled))), 'ro')
    plt.show()

    optimize_brightness(50, target, cells)
    optimize_position_and_brightness(300, target, cells, stage=1)
    optimize_pose(200, target, cells)
    optimize_brightness(100, target, cells)
    delete_superfluous(1.05, target)
    if(positions(cells)):
        split(splitting_eccentricity)
        optimize_position(100, target, cells)
        optimize_pose(200, target, cells)

        split(splitting_eccentricity)
        optimize_position(100, target, cells)
        optimize_pose(500, target, cells)

        split(splitting_eccentricity)
        optimize_position(100, target, cells)
        optimize_pose(500, target, cells)
        
        wiggled = wiggled_gradients(target)
        x, y = density_diagram(wiggled)
        plt.plot(x, y)
        plt.plot(np.array(wiggled), np.zeros((len(wiggled))), 'ro')
        plt.show()

        if createanimation:
            optimize_pose(100, target, cells)

        diff, simulated = loss_fn(cells, target, stage=2)
        if(plotit):
            plot(diff, simulated)

        save_dir = '../data/stemcells/simulated/'+ ('preprocessed_' if preprocessed else 'real_') \
            +str(width)+'x'+str(height)+'_'+str(splitting_eccentricity)+'_'+str(pos_lr)+'_'+str(pose_lr)+'_'+str(brightness_lr)+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if createanimation:
            print('creating blob animation...')
            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255)
            def animate(i):
                factor = 255.0/np.max(simulation_sequence[i])
                im.set_array(factor* simulation_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(simulation_sequence), interval=1, blit=True)
            anim.save(save_dir + 'simulated_blobs_' + str(i) +'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.show()

            print('creating blob loss animation...')
            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255)
            def animate(i):
                factor = 255.0/np.max(loss_sequence[i])
                im.set_array(factor* loss_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(loss_sequence), interval=1, blit=True)
            anim.save(save_dir + 'simulated_blobs_loss_' + str(i) +'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.show()

        pos_array = torch.stack(positions(cells))
        pose_array = torch.stack(pose_matrices(cells))
        torch.save(pos_array, save_dir+f'{i:03}'+'pos.pt')
        torch.save(pose_array, save_dir+f'{i:03}'+'pose.pt')

        if(createanimation):
            simulation_sequence = []
            loss_sequence = []
        del pose_array
        del pos_array

    del cells
    del target
    torch.cuda.empty_cache()
    i += 1
