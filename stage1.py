import pyredner
import torch
import os
import numpy as np
import cell
import csv
import imageio
import skimage.transform
import glob
from setup import width, height, granularity, device
from collections import deque
import load
from matplotlib import pyplot as plt, animation
import time
from scene import create_scene_environment, create_scene, regularizer


def run_stage(alpha, reflectance_learn_rate, vertex_learn_rate, circles, simulated_path=None, target_regex='../data/stemcells/01/*.tif'):
    """ 
        run the optimization for all images fitting the *target_regex* with the given regularization weight *alpha*,
        the given learning rates and a circles by circles grid starting layout (if simulated_path is None)
    """
    model_init_string = '_'+str(circles)+'x'+str(circles)+'circles_' + str(granularity)
    model_parameter_string = '_' + str(vertex_learn_rate) +'_' + str(reflectance_learn_rate) + '_' + str(alpha)
    simulations = None
    if simulated_path:
        simulations = load.simulated_ellipses(simulated_path)

    # load the target images into numpy arrays
    targets = deque([])
    for path in sorted(glob.glob(target_regex)):
        raw_image = imageio.imread(path)

        target = torch.from_numpy(skimage.transform.rescale(raw_image,
            (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)

        targets.append(target)


    pyredner.set_use_gpu(torch.cuda.is_available())
    # create the scene environment
    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    starttime = time.time()
    avg_final_rend_loss = 0
    avg_final_reg_loss = 0

    # perform the optimization for every target image in targets
    for j, target in enumerate(targets):
        print(circles, alpha, j)
        cells = []
        
        if simulations:
            # if a starting layout in the form of ellipses was provided, turn the ellipse representations into polygons and use those
            cells = simulations[j]
            for i, c in enumerate(cells):
                c.create_polygon(112)
                c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
        else:
            # if no ellipses were provided, start of with a *circles* by *circles* grid layout
            for i in range(width//(circles + 1), circles*width//(circles + 1) + 1, width//(circles + 1)):
                for k in range(height//(circles + 1), circles*height//(circles + 1) + 1, width//(circles + 1)):
                    pos = torch.FloatTensor([i, k]).to(device)
                    M = 2.7*torch.FloatTensor([[3.6e-02, 1.3e-05], [1.3e-05, 3.6e-02]]).to(device)
                    c = cell.Cell(M, pos, 0)

                    c.create_polygon(100)
                    c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
                    cells.append(c)
        
        _, render = create_scene(pyredner, cam, shape_light, area_lights, cells)

        optimizerref = torch.optim.Adam(cell.redner_reflectances(cells), lr=1e-2)
        reflectances = np.array([c.diffuse_reflectance.sum().item() for c in cells if c.visible])
        
        # Optimize reflectances of cells
        for t in range(30):
            print('iteration:', t)
            optimizerref.zero_grad()
            # Forward pass: render the image and compute the difference to the target
            img = render(4, 1).sum(dim=-1)
            diff = (target - img)
            if (t % 30 == 29) and plotit:
                plt.figure(1)
                plt.imshow(img.cpu().detach().numpy(), cmap='gray')
                plt.figure(2)
                plt.imshow(diff.abs().cpu().detach(), cmap='gray')
                plt.show()
            simple_loss = diff.pow(2).sum()
            loss = simple_loss
            # backprop gradients and update reflectances
            loss.backward(retain_graph=True)
            optimizerref.step()

        # pre-prune some virtual cells that clearly don't cover a real one based on how much their reflectance was adjusted just now
        for c in cells:
            print(c.vertices.grad.sum().abs()/granularity)
            if(c.vertices.grad.sum().abs()/granularity < 3e-2):
                c.visible = False
        
        _, render = create_scene(pyredner, cam, shape_light, area_lights, [c for c in cells if c.visible])

        optimizerref = torch.optim.Adam(cell.redner_reflectances(cells), lr=reflectance_learn_rate)
        optimizer = torch.optim.Adam(cell.redner_vertices(cells), lr=vertex_learn_rate)
        vertex_sequence = []
        simulation_sequence = []

        # Optimize the shape of the cells

        best_rend_loss = float('inf')
        stopping_counter = 0

        for t in range(120):
            print('vertex iteration', t)
            optimizer.zero_grad()
            optimizerref.zero_grad()
            
            # Forward pass: render the image and compute the difference to the target
            img = render(4, 1).sum(dim=-1)
            diff = (target - img)

            if saveanimation:
                vlistcolored = cell.render_vertex_list(cells, torch.max(target), target).detach().cpu()
                cpuimg = img.detach().cpu()
                vertex_sequence = vertex_sequence + [vlistcolored.numpy()]
                simulation_sequence = simulation_sequence + [cpuimg.numpy()]

            if(t % 40 == 39) and plotit:
                vlistcolored = cell.render_vertex_list(cells, torch.max(target), target).detach().cpu()
                cpuimg = img.detach().cpu()
                reflectances = np.array([c.diffuse_reflectance.sum().item() for c in cells if c.visible])

                plt.figure('rendered')
                plt.imshow(cpuimg, cmap='gray')
                plt.figure('difference')
                plt.imshow(diff.abs().cpu().detach(), cmap='gray')
                plt.figure('vertices')
                plt.imshow(vlistcolored, cmap='gray')
                plt.show()

            reg = -regularizer([c for c in cells if c.visible])
            print("regularization: ", reg)
            simple_loss = diff.pow(2).sum()
            print("simple_loss: ", simple_loss)
            # compute the loss as a combination of the "simple" (rendering) loss and the regularization loss
            loss = simple_loss + alpha * reg

            # early stopping if the rendering loss hasn't significantly improved for more than 15 iterations
            if simple_loss < best_rend_loss - 0.5:
                stopping_counter = 0
                best_rend_loss = simple_loss.item()
            else:
                stopping_counter += 1
                if stopping_counter == 15:
                    print('stopping early')
                    break
            
            # backprop gradients and update vertex positions and reflectances
            loss.backward(retain_graph=True)
            
            optimizer.step()
            optimizerref.step()

        avg_final_rend_loss += simple_loss.item()
        avg_final_reg_loss += reg.item()

        # save the resulting vertex lists and reflectances
        if not simulated_path:
            simulated_path = 'other_initilization' + model_init_string 
        target_dir = '../data/stemcells/simulated/' + simulated_path + '/optimized_vertex_lists' + model_parameter_string +  '/'

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        vertex_lists = torch.stack(cell.vertex_lists(cells))
        reflectances = torch.stack(cell.redner_reflectances(cells))
        torch.save(vertex_lists, target_dir+f'{j+1:03}'+'vert2.pt')
        torch.save(reflectances, target_dir+f'{j+1:03}'+'refl2.pt')

        # create, display and save the animations if saveanimation was set to True
        if saveanimation:
            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255, cmap='gray')
            def animate(i):
                factor = 255.0/np.max(vertex_sequence[i])
                im.set_array(factor* vertex_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(vertex_sequence), interval=1, blit=True)
            anim.save('vertex_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
            plt.show()

            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255, cmap='gray')
            def animate(i):
                factor = 255.0/np.max(simulation_sequence[i])
                im.set_array(factor* simulation_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(vertex_sequence), interval=1, blit=True)
            anim.save('simulated_cells.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
            plt.show()
    
    # Compute, print and save the average time, final regularization, and rendering loss
    timediff = time.time() - starttime
    avg_time = timediff / len(targets)
    avg_final_reg_loss /= len(targets)
    avg_final_rend_loss /= len(targets)

    print('--------------------------\n')
    print(' ', alpha, circles)
    print(' ', avg_time, avg_final_reg_loss, avg_final_rend_loss)
    print('\n--------------------------\n')

    with open(target_dir + 'info2.csv', 'w') as info:
        wr = csv.writer(info, delimiter=',')
        wr.writerow([str(avg_time), str(avg_final_rend_loss), str(avg_final_reg_loss)])


# the path from which to take the initial cell positions
# if set to None cells will start out circular in a grid layout with circles by circles many of them
simulated_path = None#'256x256_0.71_4_5e-05'
# learning rates controlling how much the reflectances and the vertex positions are adjusted each iteration step
reflectance_learn_rate = 1e-2
vertex_learn_rate = 1e-0
# if plotit is True the renderings will be plotted every few generations
plotit = False
# if saveanimation is True the development of the cells during optimization will be animated in the form of
#   a) the renderings over time
#   b) the vertex lists over time
#   c) the difference between renderings and target images over time
saveanimation = True

for circles in range(11, 13):
    for alpha in [0]:
        run_stage(alpha, reflectance_learn_rate, vertex_learn_rate, circles, simulated_path)