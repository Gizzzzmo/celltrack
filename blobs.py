import torch
import numpy as np
import os
import imageio
import glob
import skimage.transform
from cell import Cell, positions, pose_matrices, brightnesses, render_simulation
from matplotlib import pyplot as plt, animation
from setup import device, width, height, density_diagram

print(device)

# factor determines how large the initial cells are
factor = 2.8
# splitting_eccentricity determines how high an ellipse's eccentricity has to be for it to be split
splitting_eccentricity = 0.71
# learning rate for the cell positions
pos_lr = 4
# learning rate for the cell poses, i.e. their shapes and orientations
pose_lr = 5e-5
# learning rate for the cells' brightness parameters
brightness_lr = 5e-2
simulation_sequence = []
loss_sequence = []
# toggle to display and save an animation of the optimization process
createanimation = False
# toggle to display the current state of the optimization along the way
plotit = True

def createCells():
    """ create an 8x8 initial layout of circular cells """
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
    """ the loss function used """
    simulated = render_simulation(cells, stage)
    return (target-simulated), simulated

def plot(diff, simulated):
    """ plot two images (usually the current simulation and the current difference between the simulation and the target)"""
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(diff.cpu().abs().detach().numpy())
    plt.subplot(212)
    plt.imshow(simulated.cpu().detach().numpy())
    
    plt.show()

def optimize(iterations, target, optimizer, stage=2):
    """ perform *iterations* many optimization steps, with the given optimizer, and target image"""
    for i in range(iterations):
        optimizer.zero_grad()
        # compute the loss
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        # attach images to a list if an animation is to be created
        if createanimation:
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
        # backprop loss and perform optimization step
        loss.backward(retain_graph=True)
        optimizer.step()

def optimize_position(iterations, target, cells, stage=2):
    """ minimize the loss w.r.t. the cells' positions """
    print('optimizing position')
    optimizer = torch.optim.Adam(positions(cells), lr=pos_lr)
    optimize(iterations, target, optimizer, stage)

def optimize_brightness(iterations, target, cells, stage=2):
    """ minimize the loss w.r.t. the cells' brightnesses """
    print('optimizing brightness')
    optimizer = torch.optim.Adam(brightnesses(cells), lr=brightness_lr)
    optimize(iterations, target, optimizer, stage)

def optimize_pose(iterations, target, cells, stage=2):
    """ minimize the loss w.r.t. the cells' pose_matrices """
    print('optimizing pose...')
    optimizer = torch.optim.Adam(positions(cells) + pose_matrices(cells), lr=pose_lr)
    optimize(iterations, target, optimizer, stage)

def optimize_position_and_brightness(iterations, target, cells, stage=2):
    """ minimize the loss w.r.t. the cells' positions and brightnesses simultaneaously """
    optimizer1 = torch.optim.Adam(positions(cells), lr=pos_lr)
    optimizer2 = torch.optim.Adam(brightnesses(cells), lr=brightness_lr)
    for i in range(iterations):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # compute the loss
        diff, simulated = loss_fn(cells, target, stage)
        loss = diff.pow(2).sum()
        # attach images to a list if an animation is to be created
        if createanimation:
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
        # backprop loss and perform optimization step
        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()

def split(threshold):
    """ split cells into two if their eccentricity surpasses the *threshold* """
    print('splitting...')
    for cell in cells:
        # compute eccentricity via singular value decomposition of the pose matrix
        u, s, v = torch.svd(cell.pose_matrix)
        ecc = 1-(s[1]/s[0])**2
        # if the threshold is surpassed delete this cell, and create two new ones along the princial axis of the original one
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
    """ 
        delete superfluous cells by checking if the loss produced by them is at least *threshold* times greater than the norm of the image,
        or if they simply don't contribute enough to the simulation
    """
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
    """ compute how much the gradients change when one slightly changes any cell's position """
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
target_dir = '../data/stemcells/01/*.tif'
print("go")

# perform the optimization for every image in the target directory
for path in sorted(glob.glob(target_dir)):
    print(path)
    if(i < 1):
        i += 1
        continue
    # load the image
    raw_image = imageio.imread(path)
    
    # downscale the image and transform it into a pytorch tensor
    target = 255*torch.from_numpy(skimage.transform.rescale(raw_image,
        (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)
    # create intial cells
    cells = createCells()
    
    if plotit:
        wiggled = wiggled_gradients(target)
        x, y = density_diagram(wiggled)
        plt.plot(x, y)
        plt.plot(np.array(wiggled), np.zeros((len(wiggled))), 'ro')
        plt.show()

    # taking turns, optimize the brightness, the positions and the pose_matrices of all the cells
    optimize_brightness(50, target, cells)
    optimize_position_and_brightness(300, target, cells, stage=1)
    optimize_pose(200, target, cells)
    optimize_brightness(100, target, cells)
    # delete superfluous cells
    delete_superfluous(1.05, target)
    if(positions(cells)):
        # continue to do the same as before, while also splitting cells after adjusting their pose matrices
        split(splitting_eccentricity)
        optimize_position(100, target, cells)
        optimize_pose(200, target, cells)

        split(splitting_eccentricity)
        optimize_position(100, target, cells)
        optimize_pose(500, target, cells)

        split(splitting_eccentricity)
        optimize_position(100, target, cells)
        optimize_pose(500, target, cells)
        
        if plotit:
            wiggled = wiggled_gradients(target)
            x, y = density_diagram(wiggled)
            plt.plot(x, y)
            plt.plot(np.array(wiggled), np.zeros((len(wiggled))), 'ro')
            plt.show()

        if createanimation:
            optimize_pose(100, target, cells)

        if plotit:
            diff, simulated = loss_fn(cells, target, stage=2)
            plot(diff, simulated)

        # determine where to save the animation, as well as the array of cell positions and pose_matrices
        save_dir = '../data/stemcells/simulated/real_' \
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

        # save the positions and pose_matrices
        pos_array = torch.stack(positions(cells))
        pose_array = torch.stack(pose_matrices(cells))
        torch.save(pos_array, save_dir+f'{i:03}'+'pos.pt')
        torch.save(pose_array, save_dir+f'{i:03}'+'pose.pt')

        # reset the lists for the animation and delete arrays, so as not to run out of memory on the graphics card
        if createanimation:
            simulation_sequence = []
            loss_sequence = []
        del pose_array
        del pos_array

    del cells
    del target
    # force cuda to free unused memory on the graphics card
    torch.cuda.empty_cache()
    i += 1
